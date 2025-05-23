#include "storage/store/segment.h"

#include <algorithm>
#include <cstring>

#include "common/data_chunk/sel_vector.h"
#include "common/exception/copy.h"
#include "common/serializer/deserializer.h"
#include "common/serializer/serializer.h"
#include "common/system_config.h"
#include "common/type_utils.h"
#include "common/types/types.h"
#include "common/vector/value_vector.h"
#include "expression_evaluator/expression_evaluator.h"
#include "storage/buffer_manager/buffer_manager.h"
#include "storage/buffer_manager/memory_manager.h"
#include "storage/buffer_manager/spiller.h"
#include "storage/compression/compression.h"
#include "storage/compression/float_compression.h"
#include "storage/page_manager.h"
#include "storage/stats/column_stats.h"
#include "storage/store/column.h"
#include "storage/store/column_chunk_metadata.h"
#include "storage/store/compression_flush_buffer.h"
#include "storage/store/list_chunk_data.h"
#include "storage/store/string_chunk_data.h"
#include "storage/store/struct_chunk_data.h"

using namespace kuzu::common;
using namespace kuzu::evaluator;
using namespace kuzu::transaction;

namespace kuzu {
namespace storage {

void ChunkState::reclaimAllocatedPages(FileHandle& dataFH) const {
    const auto& entry = metadata.pageRange;
    if (entry.startPageIdx != INVALID_PAGE_IDX) {
        dataFH.getPageManager()->freePageRange(entry);
    }
    for (const auto& child : childrenStates) {
        child.reclaimAllocatedPages(dataFH);
    }
}

static std::shared_ptr<CompressionAlg> getCompression(const LogicalType& dataType,
    bool enableCompression) {
    if (!enableCompression) {
        return std::make_shared<Uncompressed>(dataType);
    }
    switch (dataType.getPhysicalType()) {
    case PhysicalTypeID::INT128: {
        return std::make_shared<IntegerBitpacking<int128_t>>();
    }
    case PhysicalTypeID::INT64: {
        return std::make_shared<IntegerBitpacking<int64_t>>();
    }
    case PhysicalTypeID::INT32: {
        return std::make_shared<IntegerBitpacking<int32_t>>();
    }
    case PhysicalTypeID::INT16: {
        return std::make_shared<IntegerBitpacking<int16_t>>();
    }
    case PhysicalTypeID::INT8: {
        return std::make_shared<IntegerBitpacking<int8_t>>();
    }
    case PhysicalTypeID::INTERNAL_ID:
    case PhysicalTypeID::UINT64: {
        return std::make_shared<IntegerBitpacking<uint64_t>>();
    }
    case PhysicalTypeID::UINT32: {
        return std::make_shared<IntegerBitpacking<uint32_t>>();
    }
    case PhysicalTypeID::UINT16: {
        return std::make_shared<IntegerBitpacking<uint16_t>>();
    }
    case PhysicalTypeID::UINT8: {
        return std::make_shared<IntegerBitpacking<uint8_t>>();
    }
    case PhysicalTypeID::FLOAT: {
        return std::make_shared<FloatCompression<float>>();
    }
    case PhysicalTypeID::DOUBLE: {
        return std::make_shared<FloatCompression<double>>();
    }
    default: {
        return std::make_shared<Uncompressed>(dataType);
    }
    }
}

Segment::Segment(MemoryManager& mm, LogicalType dataType, uint64_t capacity, bool enableCompression,
    ResidencyState residencyState, bool initializeToZero)
    : residencyState{residencyState}, dataType{std::move(dataType)},
      enableCompression{enableCompression},
      numBytesPerValue{getDataTypeSizeInChunk(this->dataType)}, capacity{capacity}, numValues{0},
      inMemoryStats() {
    initializeBuffer(this->dataType.getPhysicalType(), mm, initializeToZero);
    initializeFunction();
}

Segment::Segment(MemoryManager& mm, LogicalType dataType, bool enableCompression,
    const ColumnChunkMetadata& metadata, bool initializeToZero)
    : residencyState(ResidencyState::ON_DISK), dataType{std::move(dataType)},
      enableCompression{enableCompression},
      numBytesPerValue{getDataTypeSizeInChunk(this->dataType)}, capacity{0},
      numValues{metadata.numValues}, metadata{metadata} {
    initializeBuffer(this->dataType.getPhysicalType(), mm, initializeToZero);
    initializeFunction();
}

Segment::Segment(MemoryManager& mm, PhysicalTypeID dataType, bool enableCompression,
    const ColumnChunkMetadata& metadata, bool initializeToZero)
    : Segment(mm, LogicalType::ANY(dataType), enableCompression, metadata, initializeToZero) {}

void Segment::initializeBuffer(common::PhysicalTypeID physicalType, MemoryManager& mm,
    bool initializeToZero) {
    numBytesPerValue = getDataTypeSizeInChunk(physicalType);

    // Some columnChunks are much smaller than the 256KB minimum size used by allocateBuffer
    // Which would lead to excessive memory use, particularly in the partitioner
    buffer = mm.allocateBuffer(initializeToZero, getBufferSize(capacity));
}

void Segment::initializeFunction() {
    const auto compression = getCompression(dataType, enableCompression);
    getMetadataFunction = GetCompressionMetadata(compression, dataType);
    flushBufferFunction = initializeFlushBufferFunction(compression);
}

Segment::flush_buffer_func_t Segment::initializeFlushBufferFunction(
    std::shared_ptr<CompressionAlg> compression) const {
    switch (dataType.getPhysicalType()) {
    case PhysicalTypeID::BOOL: {
        // Since we compress into memory, storage is the same as fixed-sized
        // values, but we need to mark it as being boolean compressed.
        return uncompressedFlushBuffer;
    }
    case PhysicalTypeID::STRING:
    case PhysicalTypeID::INT64:
    case PhysicalTypeID::INT32:
    case PhysicalTypeID::INT16:
    case PhysicalTypeID::INT8:
    case PhysicalTypeID::INTERNAL_ID:
    case PhysicalTypeID::ARRAY:
    case PhysicalTypeID::LIST:
    case PhysicalTypeID::UINT64:
    case PhysicalTypeID::UINT32:
    case PhysicalTypeID::UINT16:
    case PhysicalTypeID::UINT8:
    case PhysicalTypeID::INT128: {
        return CompressedFlushBuffer(compression, dataType);
    }
    case PhysicalTypeID::DOUBLE: {
        return CompressedFloatFlushBuffer<double>(compression, dataType);
    }
    case PhysicalTypeID::FLOAT: {
        return CompressedFloatFlushBuffer<float>(compression, dataType);
    }
    default: {
        return uncompressedFlushBuffer;
    }
    }
}

void Segment::resetToEmpty() {
    KU_ASSERT(residencyState != ResidencyState::ON_DISK);
    KU_ASSERT(getBufferSize() == getBufferSize(capacity));
    memset(getData<uint8_t>(), 0x00, getBufferSize());
    numValues = 0;
    resetInMemoryStats();
}

static void updateInMemoryStats(ColumnChunkStats& stats, const ValueVector& values,
    uint64_t offset = 0, uint64_t numValues = std::numeric_limits<uint64_t>::max()) {
    const auto physicalType = values.dataType.getPhysicalType();
    const auto numValuesToCheck = std::min(numValues, values.state->getSelSize());
    stats.update(values.getData(), offset, numValuesToCheck, &values.getNullMask(), physicalType);
}

static void updateInMemoryStats(ColumnChunkStats& stats, const Segment* values, uint64_t offset = 0,
    uint64_t numValues = std::numeric_limits<uint64_t>::max()) {
    const auto physicalType = values->getDataType().getPhysicalType();
    const auto numValuesToCheck = std::min(values->getNumValues(), numValues);
    const auto nullMask = values->getNullMask();
    stats.update(values->getData(), offset, numValuesToCheck,
        nullMask ? &nullMask.value() : nullptr, physicalType);
}

MergedColumnChunkStats Segment::getMergedColumnChunkStats() const {
    const CompressionMetadata& onDiskMetadata = metadata.compMeta;
    ColumnChunkStats stats = inMemoryStats;
    const auto physicalType = getDataType().getPhysicalType();
    const bool isStorageValueType =
        common::TypeUtils::visit(physicalType, []<typename T>(T) { return StorageValueType<T>; });
    if (isStorageValueType) {
        stats.update(onDiskMetadata.min, onDiskMetadata.max, physicalType);
    }
    return MergedColumnChunkStats{stats, !nullData || nullData->haveNoNullsGuaranteed(),
        nullData && nullData->haveAllNullsGuaranteed()};
}

void Segment::updateStats(const common::ValueVector* vector, const common::SelectionView& selView) {
    if (selView.isUnfiltered()) {
        updateInMemoryStats(inMemoryStats, *vector);
    } else {
        TypeUtils::visit(
            getDataType().getPhysicalType(),
            [&]<StorageValueType T>(T) {
                auto firstValue = vector->firstNonNull<T>();
                if (!firstValue) {
                    return;
                }
                T min = *firstValue, max = *firstValue;
                auto update = [&](sel_t pos) {
                    const auto val = vector->getValue<T>(pos);
                    if (val < min) {
                        min = val;
                    } else if (val > max) {
                        max = val;
                    }
                };
                if (vector->hasNoNullsGuarantee()) {
                    selView.forEach(update);
                } else {
                    selView.forEach([&](auto pos) {
                        if (!vector->isNull(pos)) {
                            update(pos);
                        }
                    });
                }
                inMemoryStats.update(StorageValue(min), StorageValue(max),
                    getDataType().getPhysicalType());
            },
            []<typename T>(T) { static_assert(!StorageValueType<T>); });
    }
}

void Segment::resetInMemoryStats() {
    inMemoryStats.reset();
}

ColumnChunkMetadata Segment::getMetadataToFlush() const {
    KU_ASSERT(numValues <= capacity);
    StorageValue minValue = {}, maxValue = {};
    if (capacity > 0) {
        std::optional<NullMask> nullMask;
        if (nullData) {
            nullMask = nullData->getNullMask();
        }
        auto [min, max] =
            getMinMaxStorageValue(getData(), 0 /*offset*/, numValues, dataType.getPhysicalType(),
                nullMask.has_value() ? &*nullMask : nullptr, true /*valueRequiredIfUnsupported*/);
        minValue = min.value_or(StorageValue());
        maxValue = max.value_or(StorageValue());
    }
    KU_ASSERT(getBufferSize() == getBufferSize(capacity));
    return getMetadataFunction(buffer->getBuffer(), capacity, numValues, minValue, maxValue);
}

void Segment::append(ValueVector* vector, const SelectionView& selView) {
    KU_ASSERT(vector->dataType.getPhysicalType() == dataType.getPhysicalType());
    copyVectorToBuffer(vector, numValues, selView);
    numValues += selView.getSelSize();
    updateStats(vector, selView);
}

void Segment::append(const Segment* other, offset_t startPosInOtherChunk,
    uint32_t numValuesToAppend) {
    KU_ASSERT(other->dataType.getPhysicalType() == dataType.getPhysicalType());
    KU_ASSERT(numValues + numValuesToAppend <= capacity);
    memcpy(getData<uint8_t>() + numValues * numBytesPerValue,
        other->getData<uint8_t>() + startPosInOtherChunk * numBytesPerValue,
        numValuesToAppend * numBytesPerValue);
    numValues += numValuesToAppend;
    updateInMemoryStats(inMemoryStats, other, startPosInOtherChunk, numValuesToAppend);
}

void Segment::flush(FileHandle& dataFH) {
    const auto preScanMetadata = getMetadataToFlush();
    auto allocatedEntry = dataFH.getPageManager()->allocatePageRange(preScanMetadata.getNumPages());
    const auto flushedMetadata = flushBuffer(&dataFH, allocatedEntry, preScanMetadata);
    setToOnDisk(flushedMetadata);
}

// Note: This function is not setting child/null chunk data recursively.
void Segment::setToOnDisk(const ColumnChunkMetadata& otherMetadata) {
    residencyState = ResidencyState::ON_DISK;
    capacity = 0;
    // Note: We don't need to set the buffer to nullptr, as it allows Segment to be resized.
    buffer = buffer->getMemoryManager()->allocateBuffer(true, 0 /*size*/);
    this->metadata = otherMetadata;
    this->numValues = otherMetadata.numValues;
    resetInMemoryStats();
}

ColumnChunkMetadata Segment::flushBuffer(FileHandle* dataFH, const PageRange& entry,
    const ColumnChunkMetadata& otherMetadata) const {
    if (!otherMetadata.compMeta.isConstant() && getBufferSize() != 0) {
        KU_ASSERT(getBufferSize() == getBufferSize(capacity));
        return flushBufferFunction(buffer->getBuffer(), dataFH, entry, otherMetadata);
    }
    KU_ASSERT(otherMetadata.getNumPages() == 0);
    return otherMetadata;
}

uint64_t Segment::getBufferSize(uint64_t capacity_) const {
    switch (dataType.getLogicalTypeID()) {
    case LogicalTypeID::BOOL: {
        // 8 values per byte, and we need a buffer size which is a
        // multiple of 8 bytes.
        return ceil(capacity_ / 8.0 / 8.0) * 8;
    }
    default: {
        return numBytesPerValue * capacity_;
    }
    }
}

void Segment::initializeScanState(ChunkState& state, const Column* column) const {
    state.column = column;
    if (residencyState == ResidencyState::ON_DISK) {
        state.metadata = metadata;
        state.numValuesPerPage = state.metadata.compMeta.numValues(KUZU_PAGE_SIZE, dataType);

        state.column->populateExtraChunkState(state);
    }
}

void Segment::scan(ValueVector& output, offset_t offset, length_t length,
    sel_t posInOutputVector) const {
    KU_ASSERT(offset + length <= numValues);
    memcpy(output.getData() + posInOutputVector * numBytesPerValue,
        getData() + offset * numBytesPerValue, numBytesPerValue * length);
}

void Segment::lookup(offset_t offsetInChunk, ValueVector& output, sel_t posInOutputVector) const {
    KU_ASSERT(offsetInChunk < capacity);
    // TODO: Handle this in ColumnChunk
    // output.setNull(posInOutputVector, isNull(offsetInChunk));
    if (!output.isNull(posInOutputVector)) {
        memcpy(output.getData() + posInOutputVector * numBytesPerValue,
            getData() + offsetInChunk * numBytesPerValue, numBytesPerValue);
    }
}

void Segment::write(Segment* chunk, Segment* dstOffsets, RelMultiplicity multiplicity) {
    KU_ASSERT(chunk->dataType.getPhysicalType() == dataType.getPhysicalType() &&
              dstOffsets->getDataType().getPhysicalType() == PhysicalTypeID::INTERNAL_ID &&
              chunk->getNumValues() == dstOffsets->getNumValues());
    for (auto i = 0u; i < dstOffsets->getNumValues(); i++) {
        const auto dstOffset = dstOffsets->getValue<offset_t>(i);
        KU_ASSERT(dstOffset < capacity);
        memcpy(getData() + dstOffset * numBytesPerValue, chunk->getData() + i * numBytesPerValue,
            numBytesPerValue);
        numValues = dstOffset >= numValues ? dstOffset + 1 : numValues;
    }
    if (nullData || multiplicity == RelMultiplicity::ONE) {
        for (auto i = 0u; i < dstOffsets->getNumValues(); i++) {
            const auto dstOffset = dstOffsets->getValue<offset_t>(i);
            if (multiplicity == RelMultiplicity::ONE && isNull(dstOffset)) {
                throw CopyException(
                    stringFormat("Node with offset: {} can only have one neighbour due "
                                 "to the MANY-ONE/ONE-ONE relationship constraint.",
                        dstOffset));
            }
            if (nullData) {
                nullData->setNull(dstOffset, chunk->isNull(i));
            }
        }
    }
    updateInMemoryStats(inMemoryStats, chunk);
}

// NOTE: This function is only called in LocalTable right now when
// performing out-of-place committing. LIST has a different logic for
// handling out-of-place committing as it has to be slided. However,
// this is unsafe, as this function can also be used for other purposes
// later. Thus, an assertion is added at the first line.
void Segment::write(const ValueVector* vector, offset_t offsetInVector, offset_t offsetInChunk) {
    KU_ASSERT(dataType.getPhysicalType() != PhysicalTypeID::BOOL &&
              dataType.getPhysicalType() != PhysicalTypeID::LIST &&
              dataType.getPhysicalType() != PhysicalTypeID::ARRAY);
    if (nullData) {
        nullData->setNull(offsetInChunk, vector->isNull(offsetInVector));
    }
    if (offsetInChunk >= numValues) {
        numValues = offsetInChunk + 1;
    }
    if (!vector->isNull(offsetInVector)) {
        memcpy(getData() + offsetInChunk * numBytesPerValue,
            vector->getData() + offsetInVector * numBytesPerValue, numBytesPerValue);
    }
    static constexpr uint64_t numValuesToWrite = 1;
    updateInMemoryStats(inMemoryStats, *vector, offsetInVector, numValuesToWrite);
}

void Segment::write(const Segment* srcChunk, offset_t srcOffsetInChunk, offset_t dstOffsetInChunk,
    offset_t numValuesToCopy) {
    KU_ASSERT(srcChunk->dataType.getPhysicalType() == dataType.getPhysicalType());
    if ((dstOffsetInChunk + numValuesToCopy) >= numValues) {
        numValues = dstOffsetInChunk + numValuesToCopy;
    }
    memcpy(getData() + dstOffsetInChunk * numBytesPerValue,
        srcChunk->getData() + srcOffsetInChunk * numBytesPerValue,
        numValuesToCopy * numBytesPerValue);
    if (nullData) {
        KU_ASSERT(srcChunk->getNullData());
        nullData->write(srcChunk->getNullData(), srcOffsetInChunk, dstOffsetInChunk,
            numValuesToCopy);
    }
    updateInMemoryStats(inMemoryStats, srcChunk, srcOffsetInChunk, numValuesToCopy);
}

void Segment::resetNumValuesFromMetadata() {
    KU_ASSERT(residencyState == ResidencyState::ON_DISK);
    numValues = metadata.numValues;
    if (nullData) {
        nullData->resetNumValuesFromMetadata();
    }
}

void Segment::setToInMemory() {
    KU_ASSERT(residencyState == ResidencyState::ON_DISK);
    KU_ASSERT(capacity == 0 && getBufferSize() == 0);
    residencyState = ResidencyState::IN_MEMORY;
    numValues = 0;
    if (nullData) {
        nullData->setToInMemory();
    }
}

void Segment::resize(uint64_t newCapacity) {
    if (newCapacity > capacity) {
        capacity = newCapacity;
    }
    const auto numBytesAfterResize = getBufferSize(newCapacity);
    if (numBytesAfterResize > getBufferSize()) {
        auto resizedBuffer = buffer->getMemoryManager()->allocateBuffer(false, numBytesAfterResize);
        auto bufferSize = getBufferSize();
        auto resizedBufferData = resizedBuffer->getBuffer().data();
        memcpy(resizedBufferData, buffer->getBuffer().data(), bufferSize);
        memset(resizedBufferData + bufferSize, 0, numBytesAfterResize - bufferSize);
        buffer = std::move(resizedBuffer);
    }
    if (nullData) {
        nullData->resize(newCapacity);
    }
}

void Segment::resizeWithoutPreserve(uint64_t newCapacity) {
    if (newCapacity > capacity) {
        capacity = newCapacity;
    }
    const auto numBytesAfterResize = getBufferSize(newCapacity);
    if (numBytesAfterResize > getBufferSize()) {
        auto resizedBuffer = buffer->getMemoryManager()->allocateBuffer(false, numBytesAfterResize);
        buffer = std::move(resizedBuffer);
    }
    if (nullData) {
        nullData->resize(newCapacity);
    }
}

void Segment::populateWithDefaultVal(ExpressionEvaluator& defaultEvaluator, uint64_t& numValues_,
    ColumnStats* newColumnStats) {
    auto numValuesAppended = 0u;
    const auto numValuesToPopulate = numValues_;
    while (numValuesAppended < numValuesToPopulate) {
        const auto numValuesToAppend =
            std::min(DEFAULT_VECTOR_CAPACITY, numValuesToPopulate - numValuesAppended);
        defaultEvaluator.evaluate(numValuesToAppend);
        auto resultVector = defaultEvaluator.resultVector.get();
        KU_ASSERT(resultVector->state->getSelVector().getSelSize() == numValuesToAppend);
        append(resultVector, resultVector->state->getSelVector());
        if (newColumnStats) {
            newColumnStats->update(resultVector);
        }
        numValuesAppended += numValuesToAppend;
    }
}

void Segment::copyVectorToBuffer(ValueVector* vector, offset_t startPosInChunk,
    const SelectionView& selView) {
    auto bufferToWrite = buffer->getBuffer().data() + startPosInChunk * numBytesPerValue;
    KU_ASSERT(startPosInChunk + selView.getSelSize() <= capacity);
    const auto vectorDataToWriteFrom = vector->getData();
    if (nullData) {
        nullData->appendNulls(vector, selView, startPosInChunk);
    }
    if (selView.isUnfiltered()) {
        memcpy(bufferToWrite, vectorDataToWriteFrom, selView.getSelSize() * numBytesPerValue);
    } else {
        selView.forEach([&](auto pos) {
            memcpy(bufferToWrite, vectorDataToWriteFrom + pos * numBytesPerValue, numBytesPerValue);
            bufferToWrite += numBytesPerValue;
        });
    }
}

void Segment::setNumValues(uint64_t numValues_) {
    KU_ASSERT(numValues_ <= capacity);
    numValues = numValues_;
    if (nullData) {
        nullData->setNumValues(numValues_);
    }
}

bool Segment::numValuesSanityCheck() const {
    if (nullData) {
        return numValues == nullData->getNumValues();
    }
    return numValues <= capacity;
}

bool Segment::sanityCheck() const {
    if (nullData) {
        return nullData->sanityCheck() && numValuesSanityCheck();
    }
    return numValues <= capacity;
}

uint64_t Segment::getEstimatedMemoryUsage() const {
    return buffer->getBuffer().size() + (nullData ? nullData->getEstimatedMemoryUsage() : 0);
}

void Segment::serialize(Serializer& serializer) const {
    KU_ASSERT(residencyState == ResidencyState::ON_DISK);
    serializer.writeDebuggingInfo("data_type");
    dataType.serialize(serializer);
    serializer.writeDebuggingInfo("metadata");
    metadata.serialize(serializer);
    serializer.writeDebuggingInfo("enable_compression");
    serializer.write<bool>(enableCompression);
    serializer.writeDebuggingInfo("has_null");
    serializer.write<bool>(nullData != nullptr);
    if (nullData) {
        serializer.writeDebuggingInfo("null_data");
        nullData->serialize(serializer);
    }
}

std::unique_ptr<Segment> Segment::deserialize(MemoryManager& memoryManager, Deserializer& deSer) {
    std::string key;
    ColumnChunkMetadata metadata;
    bool enableCompression = false;
    bool hasNull = false;
    bool initializeToZero = true;
    deSer.validateDebuggingInfo(key, "data_type");
    const auto dataType = LogicalType::deserialize(deSer);
    deSer.validateDebuggingInfo(key, "metadata");
    metadata = decltype(metadata)::deserialize(deSer);
    deSer.validateDebuggingInfo(key, "enable_compression");
    deSer.deserializeValue<bool>(enableCompression);
    deSer.validateDebuggingInfo(key, "has_null");
    deSer.deserializeValue<bool>(hasNull);
    auto chunkData = ColumnChunkFactory::createSegment(memoryManager, dataType.copy(),
        enableCompression, metadata, hasNull, initializeToZero);

    switch (dataType.getPhysicalType()) {
    case PhysicalTypeID::STRUCT: {
        StructChunkData::deserialize(deSer, *chunkData);
    } break;
    case PhysicalTypeID::STRING: {
        StringChunkData::deserialize(deSer, *chunkData);
    } break;
    case PhysicalTypeID::ARRAY:
    case PhysicalTypeID::LIST: {
        ListChunkData::deserialize(deSer, *chunkData);
    } break;
    default: {
        // DO NOTHING.
    }
    }

    return chunkData;
}

void BoolSegment::append(ValueVector* vector, const SelectionView& selView) {
    KU_ASSERT(vector->dataType.getPhysicalType() == PhysicalTypeID::BOOL);
    for (auto i = 0u; i < selView.getSelSize(); i++) {
        const auto pos = selView[i];
        NullMask::setNull(getData<uint64_t>(), numValues + i, vector->getValue<bool>(pos));
    }
    if (nullData) {
        nullData->appendNulls(vector, selView, numValues);
    }
    numValues += selView.getSelSize();
    updateStats(vector, selView);
}

void BoolSegment::append(const Segment* other, offset_t startPosInOtherChunk,
    uint32_t numValuesToAppend) {
    NullMask::copyNullMask(other->getData<uint64_t>(), startPosInOtherChunk, getData<uint64_t>(),
        numValues, numValuesToAppend);
    if (nullData) {
        nullData->append(other->getNullData(), startPosInOtherChunk, numValuesToAppend);
    }
    numValues += numValuesToAppend;
    updateInMemoryStats(inMemoryStats, other, startPosInOtherChunk, numValuesToAppend);
}

void BoolSegment::scan(ValueVector& output, offset_t offset, length_t length,
    sel_t posInOutputVector) const {
    KU_ASSERT(offset + length <= numValues);
    if (nullData) {
        nullData->scan(output, offset, length, posInOutputVector);
    }
    for (auto i = 0u; i < length; i++) {
        output.setValue<bool>(posInOutputVector + i,
            NullMask::isNull(getData<uint64_t>(), offset + i));
    }
}

void BoolSegment::lookup(offset_t offsetInChunk, ValueVector& output,
    sel_t posInOutputVector) const {
    KU_ASSERT(offsetInChunk < capacity);
    output.setNull(posInOutputVector, nullData->isNull(offsetInChunk));
    if (!output.isNull(posInOutputVector)) {
        output.setValue<bool>(posInOutputVector,
            NullMask::isNull(getData<uint64_t>(), offsetInChunk));
    }
}

void BoolSegment::write(Segment* chunk, Segment* dstOffsets, RelMultiplicity) {
    KU_ASSERT(chunk->getDataType().getPhysicalType() == PhysicalTypeID::BOOL &&
              dstOffsets->getDataType().getPhysicalType() == PhysicalTypeID::INTERNAL_ID &&
              chunk->getNumValues() == dstOffsets->getNumValues());
    for (auto i = 0u; i < dstOffsets->getNumValues(); i++) {
        const auto dstOffset = dstOffsets->getValue<offset_t>(i);
        KU_ASSERT(dstOffset < capacity);
        NullMask::setNull(getData<uint64_t>(), dstOffset, chunk->getValue<bool>(i));
        if (nullData) {
            nullData->setNull(dstOffset, chunk->getNullData()->isNull(i));
        }
        numValues = dstOffset >= numValues ? dstOffset + 1 : numValues;
    }
    updateInMemoryStats(inMemoryStats, chunk);
}

void BoolSegment::write(const ValueVector* vector, offset_t offsetInVector,
    offset_t offsetInChunk) {
    KU_ASSERT(vector->dataType.getPhysicalType() == PhysicalTypeID::BOOL);
    KU_ASSERT(offsetInChunk < capacity);
    const auto valueToSet = vector->getValue<bool>(offsetInVector);
    setValue(valueToSet, offsetInChunk);
    if (nullData) {
        nullData->write(vector, offsetInVector, offsetInChunk);
    }
    numValues = offsetInChunk >= numValues ? offsetInChunk + 1 : numValues;
    if (!vector->isNull(offsetInVector)) {
        inMemoryStats.update(StorageValue{valueToSet}, dataType.getPhysicalType());
    }
}

void BoolSegment::write(const Segment* srcChunk, offset_t srcOffsetInChunk,
    offset_t dstOffsetInChunk, offset_t numValuesToCopy) {
    if (nullData) {
        nullData->write(srcChunk->getNullData(), srcOffsetInChunk, dstOffsetInChunk,
            numValuesToCopy);
    }
    if ((dstOffsetInChunk + numValuesToCopy) >= numValues) {
        numValues = dstOffsetInChunk + numValuesToCopy;
    }
    NullMask::copyNullMask(srcChunk->getData<uint64_t>(), srcOffsetInChunk, getData<uint64_t>(),
        dstOffsetInChunk, numValuesToCopy);
    updateInMemoryStats(inMemoryStats, srcChunk, srcOffsetInChunk, numValuesToCopy);
}

void InternalIDSegment::append(ValueVector* vector, const SelectionView& selView) {
    switch (vector->dataType.getPhysicalType()) {
    case PhysicalTypeID::INTERNAL_ID: {
        copyVectorToBuffer(vector, numValues, selView);
    } break;
    case PhysicalTypeID::INT64: {
        copyInt64VectorToBuffer(vector, numValues, selView);
    } break;
    default: {
        KU_UNREACHABLE;
    }
    }
    numValues += selView.getSelSize();
}

void InternalIDSegment::copyVectorToBuffer(ValueVector* vector, offset_t startPosInChunk,
    const SelectionView& selView) {
    KU_ASSERT(vector->dataType.getPhysicalType() == PhysicalTypeID::INTERNAL_ID);
    const auto relIDsInVector = reinterpret_cast<internalID_t*>(vector->getData());
    if (commonTableID == INVALID_TABLE_ID) {
        commonTableID = relIDsInVector[selView[0]].tableID;
    }
    for (auto i = 0u; i < selView.getSelSize(); i++) {
        const auto pos = selView[i];
        if (vector->isNull(pos)) {
            continue;
        }
        KU_ASSERT(relIDsInVector[pos].tableID == commonTableID);
        memcpy(getData() + (startPosInChunk + i) * numBytesPerValue, &relIDsInVector[pos].offset,
            numBytesPerValue);
    }
}

void InternalIDSegment::copyInt64VectorToBuffer(ValueVector* vector, offset_t startPosInChunk,
    const SelectionView& selView) const {
    KU_ASSERT(vector->dataType.getPhysicalType() == PhysicalTypeID::INT64);
    for (auto i = 0u; i < selView.getSelSize(); i++) {
        const auto pos = selView[i];
        if (vector->isNull(pos)) {
            continue;
        }
        memcpy(getData() + (startPosInChunk + i) * numBytesPerValue,
            &vector->getValue<offset_t>(pos), numBytesPerValue);
    }
}

void InternalIDSegment::scan(ValueVector& output, offset_t offset, length_t length,
    sel_t posInOutputVector) const {
    KU_ASSERT(offset + length <= numValues);
    KU_ASSERT(commonTableID != INVALID_TABLE_ID);
    internalID_t relID;
    relID.tableID = commonTableID;
    for (auto i = 0u; i < length; i++) {
        relID.offset = getValue<offset_t>(offset + i);
        output.setValue<internalID_t>(posInOutputVector + i, relID);
    }
}

void InternalIDSegment::lookup(offset_t offsetInChunk, ValueVector& output,
    sel_t posInOutputVector) const {
    KU_ASSERT(offsetInChunk < capacity);
    internalID_t relID;
    relID.offset = getValue<offset_t>(offsetInChunk);
    KU_ASSERT(commonTableID != INVALID_TABLE_ID);
    relID.tableID = commonTableID;
    output.setValue<internalID_t>(posInOutputVector, relID);
}

void InternalIDSegment::write(const ValueVector* vector, offset_t offsetInVector,
    offset_t offsetInChunk) {
    KU_ASSERT(vector->dataType.getPhysicalType() == PhysicalTypeID::INTERNAL_ID);
    const auto relIDsInVector = reinterpret_cast<internalID_t*>(vector->getData());
    if (commonTableID == INVALID_TABLE_ID) {
        commonTableID = relIDsInVector[offsetInVector].tableID;
    }
    KU_ASSERT(commonTableID == relIDsInVector[offsetInVector].tableID);
    if (!vector->isNull(offsetInVector)) {
        memcpy(getData() + offsetInChunk * numBytesPerValue, &relIDsInVector[offsetInVector].offset,
            numBytesPerValue);
    }
    if (offsetInChunk >= numValues) {
        numValues = offsetInChunk + 1;
    }
}

void InternalIDSegment::append(const Segment* other, offset_t startPosInOtherChunk,
    uint32_t numValuesToAppend) {
    Segment::append(other, startPosInOtherChunk, numValuesToAppend);
    commonTableID = other->cast<InternalIDSegment>().commonTableID;
}

std::optional<NullMask> Segment::getNullMask() const {
    return nullData ? std::optional(nullData->getNullMask()) : std::nullopt;
}

std::unique_ptr<Segment> ColumnChunkFactory::createSegment(MemoryManager& mm, LogicalType dataType,
    bool enableCompression, uint64_t capacity, ResidencyState residencyState, bool hasNullData,
    bool initializeToZero) {
    switch (dataType.getPhysicalType()) {
    case PhysicalTypeID::BOOL: {
        return std::make_unique<BoolSegment>(mm, capacity, enableCompression, residencyState,
            hasNullData);
    }
    case PhysicalTypeID::INT64:
    case PhysicalTypeID::INT32:
    case PhysicalTypeID::INT16:
    case PhysicalTypeID::INT8:
    case PhysicalTypeID::UINT64:
    case PhysicalTypeID::UINT32:
    case PhysicalTypeID::UINT16:
    case PhysicalTypeID::UINT8:
    case PhysicalTypeID::INT128:
    case PhysicalTypeID::DOUBLE:
    case PhysicalTypeID::FLOAT:
    case PhysicalTypeID::INTERVAL: {
        return std::make_unique<Segment>(mm, std::move(dataType), capacity, enableCompression,
            residencyState, hasNullData, initializeToZero);
    }
    case PhysicalTypeID::INTERNAL_ID: {
        return std::make_unique<InternalIDSegment>(mm, capacity, enableCompression, residencyState);
    }
    case PhysicalTypeID::STRING: {
        return std::make_unique<StringChunkData>(mm, std::move(dataType), capacity,
            enableCompression, residencyState);
    }
    case PhysicalTypeID::ARRAY:
    case PhysicalTypeID::LIST: {
        return std::make_unique<ListChunkData>(mm, std::move(dataType), capacity, enableCompression,
            residencyState);
    }
    case PhysicalTypeID::STRUCT: {
        return std::make_unique<StructChunkData>(mm, std::move(dataType), capacity,
            enableCompression, residencyState);
    }
    default:
        KU_UNREACHABLE;
    }
}

std::unique_ptr<Segment> ColumnChunkFactory::createSegment(MemoryManager& mm, LogicalType dataType,
    bool enableCompression, ColumnChunkMetadata& metadata, bool hasNullData,
    bool initializeToZero) {
    switch (dataType.getPhysicalType()) {
    case PhysicalTypeID::BOOL: {
        return std::make_unique<BoolSegment>(mm, enableCompression, metadata, hasNullData);
    }
    case PhysicalTypeID::INT64:
    case PhysicalTypeID::INT32:
    case PhysicalTypeID::INT16:
    case PhysicalTypeID::INT8:
    case PhysicalTypeID::UINT64:
    case PhysicalTypeID::UINT32:
    case PhysicalTypeID::UINT16:
    case PhysicalTypeID::UINT8:
    case PhysicalTypeID::INT128:
    case PhysicalTypeID::DOUBLE:
    case PhysicalTypeID::FLOAT:
    case PhysicalTypeID::INTERVAL: {
        return std::make_unique<Segment>(mm, std::move(dataType), enableCompression, metadata,
            hasNullData, initializeToZero);
    }
        // Physically, we only materialize offset of INTERNAL_ID, which is same as INT64,
    case PhysicalTypeID::INTERNAL_ID: {
        // INTERNAL_ID should never have nulls.
        return std::make_unique<InternalIDSegment>(mm, enableCompression, metadata);
    }
    /* TODO: Custom string segment, and move ListChunkData and StructChunkDatao to
     * ListChunk/StructChunk */
    case PhysicalTypeID::STRING: {
        return std::make_unique<StringChunkData>(mm, enableCompression, metadata);
    }
    /*
    case PhysicalTypeID::ARRAY:
    case PhysicalTypeID::LIST: {
        return std::make_unique<ListChunkData>(mm, std::move(dataType), enableCompression,
            metadata);
    }
    case PhysicalTypeID::STRUCT: {
        return std::make_unique<StructChunkData>(mm, std::move(dataType), enableCompression,
            metadata);
    }
    */
    default:
        KU_UNREACHABLE;
    }
}

MemoryManager& Segment::getMemoryManager() const {
    return *buffer->getMemoryManager();
}

uint8_t* Segment::getData() const {
    return buffer->getBuffer().data();
}
uint64_t Segment::getBufferSize() const {
    return buffer->getBuffer().size_bytes();
}

void Segment::loadFromDisk() {
    buffer->getMemoryManager()->getBufferManager()->getSpillerOrSkip(
        [&](auto& spiller) { spiller.loadFromDisk(*this); });
}

uint64_t Segment::spillToDisk() {
    uint64_t spilledBytes = 0;
    buffer->getMemoryManager()->getBufferManager()->getSpillerOrSkip(
        [&](auto& spiller) { spilledBytes = spiller.spillToDisk(*this); });
    return spilledBytes;
}

void Segment::reclaimStorage(FileHandle& dataFH) {
    if (residencyState == ResidencyState::ON_DISK) {
        if (metadata.getStartPageIdx() != INVALID_PAGE_IDX) {
            dataFH.getPageManager()->freePageRange(metadata.pageRange);
        }
    }
}

Segment::~Segment() = default;

} // namespace storage
} // namespace kuzu

#pragma once

#include <functional>
#include <variant>

#include "common/data_chunk/sel_vector.h"
#include "common/enums/rel_multiplicity.h"
#include "common/null_mask.h"
#include "common/system_config.h"
#include "common/types/types.h"
#include "common/vector/value_vector.h"
#include "storage/compression/compression.h"
#include "storage/enums/residency_state.h"
#include "storage/store/column_chunk_metadata.h"
#include "storage/store/column_chunk_stats.h"
#include "storage/store/column_reader_writer.h"
#include "storage/store/in_memory_exception_chunk.h"

namespace kuzu {
namespace evaluator {
class ExpressionEvaluator;
} // namespace evaluator

namespace transaction {
class Transaction;
} // namespace transaction

namespace storage {

class Column;
class ColumnStats;
class FileHandle;

// TODO(bmwinger): Hide access to variables.
struct SegmentState {
    const Column* column;
    ColumnChunkMetadata metadata;
    uint64_t numValuesPerPage = UINT64_MAX;

    // TODO: Move to ChunkState
    // Used for struct/list/string columns.
    std::vector<SegmentState> childrenStates;

    // Used for floating point columns
    std::variant<std::unique_ptr<InMemoryExceptionChunk<double>>,
        std::unique_ptr<InMemoryExceptionChunk<float>>>
        alpExceptionChunk;

    SegmentState() : column{nullptr} {}
    SegmentState(ColumnChunkMetadata metadata, uint64_t numValuesPerPage)
        : column{nullptr}, metadata{std::move(metadata)}, numValuesPerPage{numValuesPerPage} {}

    SegmentState& getChildState(common::idx_t childIdx) {
        KU_ASSERT(childIdx < childrenStates.size());
        return childrenStates[childIdx];
    }
    const SegmentState& getChildState(common::idx_t childIdx) const {
        KU_ASSERT(childIdx < childrenStates.size());
        return childrenStates[childIdx];
    }

    template<std::floating_point T>
    InMemoryExceptionChunk<T>* getExceptionChunk() {
        using GetType = std::unique_ptr<InMemoryExceptionChunk<T>>;
        KU_ASSERT(std::holds_alternative<GetType>(alpExceptionChunk));
        return std::get<GetType>(alpExceptionChunk).get();
    }

    template<std::floating_point T>
    const InMemoryExceptionChunk<T>* getExceptionChunkConst() const {
        using GetType = std::unique_ptr<InMemoryExceptionChunk<T>>;
        KU_ASSERT(std::holds_alternative<GetType>(alpExceptionChunk));
        return std::get<GetType>(alpExceptionChunk).get();
    }

    void reclaimAllocatedPages(FileHandle& dataFH) const;
};

struct ChunkState {
    std::vector<SegmentState> segmentStates;
    std::unique_ptr<ChunkState> nullState;
};

class Spiller;
// Base data segment covers all fixed-sized data types.
class KUZU_API Segment {
public:
    friend struct ColumnChunkFactory;
    // For spilling to disk we need access to the underlying buffer
    friend class Spiller;

    Segment(MemoryManager& mm, common::LogicalType dataType, uint64_t capacity,
        bool enableCompression, ResidencyState residencyState, bool initializeToZero = true);
    Segment(MemoryManager& mm, common::LogicalType dataType, bool enableCompression,
        const ColumnChunkMetadata& metadata, bool initializeToZero = true);
    Segment(MemoryManager& mm, common::PhysicalTypeID physicalType, bool enableCompression,
        const ColumnChunkMetadata& metadata, bool initializeToZero = true);
    virtual ~Segment();

    template<typename T>
    T getValue(common::offset_t pos) const {
        KU_ASSERT(pos < numValues);
        KU_ASSERT(residencyState != ResidencyState::ON_DISK);
        return getData<T>()[pos];
    }
    template<typename T>
    void setValue(T val, common::offset_t pos) {
        KU_ASSERT(pos < capacity);
        KU_ASSERT(residencyState != ResidencyState::ON_DISK);
        getData<T>()[pos] = val;
        if (pos >= numValues) {
            numValues = pos + 1;
        }
        if constexpr (StorageValueType<T>) {
            inMemoryStats.update(StorageValue{val}, dataType.getPhysicalType());
        }
    }

    common::LogicalType& getDataType() { return dataType; }
    const common::LogicalType& getDataType() const { return dataType; }
    ResidencyState getResidencyState() const { return residencyState; }
    bool isCompressionEnabled() const { return enableCompression; }
    ColumnChunkMetadata& getMetadata() {
        KU_ASSERT(residencyState == ResidencyState::ON_DISK);
        return metadata;
    }
    const ColumnChunkMetadata& getMetadata() const {
        KU_ASSERT(residencyState == ResidencyState::ON_DISK);
        return metadata;
    }
    void setMetadata(const ColumnChunkMetadata& metadata_) {
        KU_ASSERT(residencyState == ResidencyState::ON_DISK);
        metadata = metadata_;
    }

    virtual void resetToEmpty();

    // Note that the startPageIdx is not known, so it will always be common::INVALID_PAGE_IDX
    virtual ColumnChunkMetadata getMetadataToFlush() const;

    virtual void append(common::ValueVector* vector, const common::SelectionView& selView);
    virtual void append(const Segment* other, common::offset_t startPosInOtherChunk,
        uint32_t numValuesToAppend);

    virtual void flush(FileHandle& dataFH);

    ColumnChunkMetadata flushBuffer(FileHandle* dataFH, const PageRange& entry,
        const ColumnChunkMetadata& metadata) const;

    static common::page_idx_t getNumPagesForBytes(uint64_t numBytes) {
        return (numBytes + common::KUZU_PAGE_SIZE - 1) / common::KUZU_PAGE_SIZE;
    }

    uint64_t getNumBytesPerValue() const { return numBytesPerValue; }
    uint8_t* getData() const;
    template<typename T>
    T* getData() const {
        return reinterpret_cast<T*>(getData());
    }
    uint64_t getBufferSize() const;

    virtual void initializeScanState(ChunkState& state, const Column* column) const;
    virtual void scan(common::ValueVector& output, common::offset_t offset, common::length_t length,
        common::sel_t posInOutputVector = 0) const;
    virtual void lookup(common::offset_t offsetInChunk, common::ValueVector& output,
        common::sel_t posInOutputVector) const;

    // TODO(Guodong): In general, this is not a good interface. Instead of passing in
    // `offsetInVector`, we should flatten the vector to pos at `offsetInVector`.
    virtual void write(const common::ValueVector* vector, common::offset_t offsetInVector,
        common::offset_t offsetInChunk);
    virtual void write(Segment* segment, Segment* offsetsInSegment,
        common::RelMultiplicity multiplicity);
    virtual void write(const Segment* srcSegment, common::offset_t srcOffsetInSegment,
        common::offset_t dstOffsetInSegment, common::offset_t numValuesToCopy);

    virtual void setToInMemory();
    // numValues must be at least the number of values the ColumnChunk was first initialized
    // with
    // reverse data and zero the part exceeding the original size
    virtual void resize(uint64_t newCapacity);
    // the opposite of the resize method, just simple resize
    virtual void resizeWithoutPreserve(uint64_t newCapacity);

    void populateWithDefaultVal(evaluator::ExpressionEvaluator& defaultEvaluator,
        uint64_t& numValues_, ColumnStats* newColumnStats);
    virtual void finalize() {
        KU_ASSERT(residencyState != ResidencyState::ON_DISK);
        // DO NOTHING.
    }

    uint64_t getCapacity() const { return capacity; }
    uint64_t getNumValues() const { return numValues; }
    // TODO(Guodong): Alternatively, we can let `getNumValues` read from metadata when ON_DISK.
    virtual void resetNumValuesFromMetadata();
    virtual void setNumValues(uint64_t numValues_);
    virtual void syncNumValues() {}
    virtual bool numValuesSanityCheck() const;

    virtual bool sanityCheck() const;

    virtual uint64_t getEstimatedMemoryUsage() const;

    virtual void serialize(common::Serializer& serializer) const;
    static std::unique_ptr<ColumnChunkData> deserialize(MemoryManager& mm,
        common::Deserializer& deSer);

    template<typename TARGET>
    TARGET& cast() {
        return common::ku_dynamic_cast<TARGET&>(*this);
    }
    template<typename TARGET>
    const TARGET& cast() const {
        return common::ku_dynamic_cast<const TARGET&>(*this);
    }
    MemoryManager& getMemoryManager() const;

    void loadFromDisk();
    uint64_t spillToDisk();

    MergedColumnChunkStats getMergedColumnChunkStats() const;

    void updateStats(const common::ValueVector* vector, const common::SelectionView& selVector);

    virtual void reclaimStorage(FileHandle& dataFH);

protected:
    // Initializes the data buffer and functions. They are (and should be) only called in
    // constructor.
    void initializeBuffer(common::PhysicalTypeID physicalType, MemoryManager& mm,
        bool initializeToZero);
    void initializeFunction();

    // Note: This function is not setting child/null chunk data recursively.
    void setToOnDisk(const ColumnChunkMetadata& metadata);

    virtual void copyVectorToBuffer(common::ValueVector* vector, common::offset_t startPosInChunk,
        const common::SelectionView& selView);

    void resetInMemoryStats();

private:
    using flush_buffer_func_t = std::function<ColumnChunkMetadata(const std::span<uint8_t>,
        FileHandle*, const PageRange&, const ColumnChunkMetadata&)>;
    flush_buffer_func_t initializeFlushBufferFunction(
        std::shared_ptr<CompressionAlg> compression) const;
    uint64_t getBufferSize(uint64_t capacity_) const;

protected:
    using get_metadata_func_t = std::function<ColumnChunkMetadata(const std::span<uint8_t>,
        uint64_t, uint64_t, StorageValue, StorageValue)>;
    using get_min_max_func_t =
        std::function<std::pair<StorageValue, StorageValue>(const uint8_t*, uint64_t)>;

    ResidencyState residencyState;
    common::LogicalType dataType;
    bool enableCompression;
    uint32_t numBytesPerValue;
    uint64_t capacity;
    std::unique_ptr<MemoryBuffer> buffer;
    uint64_t numValues;
    flush_buffer_func_t flushBufferFunction;
    get_metadata_func_t getMetadataFunction;

    // On-disk metadata for column chunk.
    ColumnChunkMetadata metadata;

    // Stats for any in-memory updates applied to the column chunk
    // This will be merged with the on-disk metadata to get the overall stats
    ColumnChunkStats inMemoryStats;
};

template<>
inline void Segment::setValue(bool val, common::offset_t pos) {
    // Buffer is rounded up to the nearest 8 bytes so that this cast is safe
    common::NullMask::setNull(getData<uint64_t>(), pos, val);
}

template<>
inline bool Segment::getValue(common::offset_t pos) const {
    // Buffer is rounded up to the nearest 8 bytes so that this cast is safe
    return common::NullMask::isNull(getData<uint64_t>(), pos);
}

// Stored as bitpacked booleans in-memory and on-disk
class BoolSegment : public Segment {
public:
    BoolSegment(MemoryManager& mm, uint64_t capacity, bool enableCompression, ResidencyState type)
        : Segment(mm, common::LogicalType::BOOL(), capacity,
              // Booleans are always bitpacked, but this can also enable constant compression
              enableCompression, type, true) {}
    BoolSegment(MemoryManager& mm, bool enableCompression, const ColumnChunkMetadata& metadata)
        : Segment{mm, common::LogicalType::BOOL(), enableCompression, metadata, true} {}

    void append(common::ValueVector* vector, const common::SelectionView& sel) final;
    void append(const Segment* other, common::offset_t startPosInOtherChunk,
        uint32_t numValuesToAppend) override;

    void scan(common::ValueVector& output, common::offset_t offset, common::length_t length,
        common::sel_t posInOutputVector = 0) const override;
    void lookup(common::offset_t offsetInChunk, common::ValueVector& output,
        common::sel_t posInOutputVector) const override;

    void write(const common::ValueVector* vector, common::offset_t offsetInVector,
        common::offset_t offsetInChunk) override;
    void write(Segment* segment, Segment* dstOffsets, common::RelMultiplicity multiplicity) final;
    void write(const Segment* srcSegment, common::offset_t srcOffsetInSegment,
        common::offset_t dstOffsetInSegment, common::offset_t numValuesToCopy) override;
};

class KUZU_API InternalIDSegment final : public Segment {
public:
    // Physically, we only materialize offset of INTERNAL_ID, which is same as UINT64,
    InternalIDSegment(MemoryManager& mm, uint64_t capacity, bool enableCompression,
        ResidencyState residencyState)
        : Segment(mm, common::LogicalType::INTERNAL_ID(), capacity, enableCompression,
              residencyState),
          commonTableID{common::INVALID_TABLE_ID} {}
    InternalIDSegment(MemoryManager& mm, bool enableCompression,
        const ColumnChunkMetadata& metadata)
        : Segment{mm, common::LogicalType::INTERNAL_ID(), enableCompression, metadata,
              },
          commonTableID{common::INVALID_TABLE_ID} {}

    void append(common::ValueVector* vector, const common::SelectionView& selView) override;

    void copyVectorToBuffer(common::ValueVector* vector, common::offset_t startPosInChunk,
        const common::SelectionView& selView) override;

    void copyInt64VectorToBuffer(common::ValueVector* vector, common::offset_t startPosInChunk,
        const common::SelectionView& selView) const;

    void scan(common::ValueVector& output, common::offset_t offset, common::length_t length,
        common::sel_t posInOutputVector = 0) const override;
    void lookup(common::offset_t offsetInChunk, common::ValueVector& output,
        common::sel_t posInOutputVector) const override;

    void write(const common::ValueVector* vector, common::offset_t offsetInVector,
        common::offset_t offsetInChunk) override;

    void append(const Segment* other, common::offset_t startPosInOtherChunk,
        uint32_t numValuesToAppend) override;

    void setTableID(common::table_id_t tableID) { commonTableID = tableID; }
    common::table_id_t getTableID() const { return commonTableID; }

    common::offset_t operator[](common::offset_t pos) const {
        return getValue<common::offset_t>(pos);
    }
    common::offset_t& operator[](common::offset_t pos) { return getData<common::offset_t>()[pos]; }

private:
    common::table_id_t commonTableID;
};

struct SegmentFactory {
    static std::unique_ptr<Segment> createSegment(MemoryManager& mm, common::LogicalType dataType,
        bool enableCompression, uint64_t capacity, ResidencyState residencyState,
        bool initializeToZero = true);
    static std::unique_ptr<Segment> createSegment(MemoryManager& mm, common::LogicalType dataType,
        bool enableCompression, ColumnChunkMetadata& metadata, bool initializeToZero);
};

} // namespace storage
} // namespace kuzu

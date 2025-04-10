#include "index/hnsw_graph.h"

#include "storage/local_cached_column.h"
#include "storage/store/list_chunk_data.h"
#include "storage/store/node_table.h"
#include "storage/store/rel_table.h"
#include "transaction/transaction.h"

using namespace kuzu::storage;

namespace kuzu {
namespace vector_extension {

EmbeddingScanState::EmbeddingScanState(const transaction::Transaction* transaction,
    MemoryManager* mm, NodeTable& nodeTable, common::column_id_t columnID) {
    std::vector columnIDs{columnID};
    // The first ValueVector in scanChunk is reserved for nodeIDs.
    std::vector<common::LogicalType> types;
    types.emplace_back(common::LogicalType::INTERNAL_ID());
    types.emplace_back(nodeTable.getColumn(columnID).getDataType().copy());
    scanChunk = Table::constructDataChunk(mm, std::move(types));
    std::vector outVectors{&scanChunk.getValueVectorMutable(1)};
    scanState = std::make_unique<NodeTableScanState>(&scanChunk.getValueVectorMutable(0),
        outVectors, scanChunk.state);
    scanState->setToTable(transaction, &nodeTable, std::move(columnIDs));
}

EmbeddingColumn::EmbeddingColumn(transaction::Transaction* transaction, EmbeddingTypeInfo typeInfo,
    NodeTable& nodeTable, common::column_id_t columnID)
    : typeInfo{std::move(typeInfo)}, cachedData{nullptr}, nodeTable{nodeTable} {
    auto& cacheManager = transaction->getLocalCacheManager();
    const auto key = CachedColumn::getKey(nodeTable.getTableID(), columnID);
    if (cacheManager.contains(key)) {
        cachedData = transaction->getLocalCacheManager().at(key).cast<CachedColumn>();
    }
}

float* EmbeddingColumn::getEmbedding(transaction::Transaction* transaction,
    NodeTableScanState* scanState, common::offset_t offset) const {
    if (cachedData) {
        auto [nodeGroupIdx, offsetInGroup] = StorageUtils::getNodeGroupIdxAndOffsetInChunk(offset);
        KU_ASSERT(nodeGroupIdx < cachedData->columnChunks.size());
        const auto& listChunk = cachedData->columnChunks[nodeGroupIdx]->cast<ListChunkData>();
        return &listChunk.getDataColumnChunk()
                    ->getData<float>()[listChunk.getListStartOffset(offsetInGroup)];
    }
    scan(transaction, scanState, offset);
    KU_ASSERT(scanState->outputVectors.size() == 1 &&
              scanState->outputVectors[0]->state->getSelVector()[0] == 0);
    const auto value = scanState->outputVectors[0]->getValue<common::list_entry_t>(0);
    KU_ASSERT(value.size == typeInfo.dimension);
    KU_UNUSED(value);
    const auto dataVector = common::ListVector::getDataVector(scanState->outputVectors[0]);
    return reinterpret_cast<float*>(dataVector->getData()) + value.offset;
}

std::vector<float*> EmbeddingColumn::getEmbeddings(transaction::Transaction* transaction,
    NodeTableScanState* scanState, const std::vector<common::offset_t>& offsets) const {
    std::vector<float*> embeddings;
    embeddings.reserve(offsets.size());
    if (cachedData) {
        for (const auto& offset : offsets) {
            auto [nodeGroupIdx, offsetInGroup] =
                StorageUtils::getNodeGroupIdxAndOffsetInChunk(offset);
            KU_ASSERT(nodeGroupIdx < cachedData->columnChunks.size());
            const auto& listChunk = cachedData->columnChunks[nodeGroupIdx]->cast<ListChunkData>();
            embeddings.push_back(&listChunk.getDataColumnChunk()
                    ->getData<float>()[listChunk.getListStartOffset(offsetInGroup)]);
        }
    } else {
        scanState->nodeIDVector->state->getSelVectorUnsafe().setToUnfiltered(offsets.size());
        for (auto i = 0u; i < offsets.size(); ++i) {
            scanState->nodeIDVector->setValue(i,
                common::internalID_t{offsets[i], nodeTable.getTableID()});
        }
        scanState->source = TableScanSource::COMMITTED;
        [[maybe_unused]] const auto result = nodeTable.lookupMultiple(transaction, *scanState);
        KU_ASSERT(
            scanState->outputVectors.size() == 1 &&
            scanState->outputVectors[0]->state->getSelVector().getSelSize() == offsets.size());
        const auto dataVector = common::ListVector::getDataVector(scanState->outputVectors[0]);
        for (auto i = 0u; i < offsets.size(); ++i) {
            const auto val = scanState->outputVectors[0]->getValue<common::list_entry_t>(i);
            embeddings.push_back(reinterpret_cast<float*>(dataVector->getData() + val.offset));
        }
    }
    return embeddings;
}

bool EmbeddingColumn::isNull(transaction::Transaction* transaction, NodeTableScanState* scanState,
    common::offset_t offset) const {
    auto [nodeGroupIdx, offsetInGroup] = StorageUtils::getNodeGroupIdxAndOffsetInChunk(offset);
    if (cachedData) {
        KU_ASSERT(nodeGroupIdx < cachedData->columnChunks.size());
        const auto& listChunk = cachedData->columnChunks[nodeGroupIdx]->cast<ListChunkData>();
        return listChunk.isNull(offsetInGroup);
    }
    scanState->outputVectors[0]->setAllNonNull();
    scan(transaction, scanState, offset);
    KU_ASSERT(scanState->nodeIDVector->state->getSelVector().getSelSize() == 1);
    const auto pos = scanState->nodeIDVector->state->getSelVector()[0];
    return scanState->outputVectors[0]->isNull(pos);
}

void EmbeddingColumn::scan(transaction::Transaction* transaction, NodeTableScanState* scanState,
    common::offset_t offset) const {
    KU_ASSERT(scanState != nullptr);
    scanState->nodeIDVector->setValue(0, common::internalID_t{offset, nodeTable.getTableID()});
    scanState->nodeIDVector->state->getSelVectorUnsafe().setToUnfiltered(1);
    scanState->source = TableScanSource::COMMITTED;
    scanState->nodeGroupIdx = StorageUtils::getNodeGroupIdx(offset);
    nodeTable.initScanState(transaction, *scanState);
    [[maybe_unused]] const auto result = nodeTable.lookup(transaction, *scanState);
}

namespace {
template<std::integral CompressedType>
struct TypedCompressedView final : CompressedOffsetsView {
    explicit TypedCompressedView(const uint8_t* data, common::offset_t numEntries)
        : dstNodes(reinterpret_cast<std::atomic<CompressedType>*>(const_cast<uint8_t*>(data)),
              numEntries),
          invalidOffset{std::numeric_limits<CompressedType>::max()} {}

    common::offset_t getNodeOffsetAtomic(common::offset_t idx) const override {
        return dstNodes[idx].load(std::memory_order_relaxed);
    }

    void setNodeOffsetAtomic(common::offset_t idx, common::offset_t nodeOffset) override {
        KU_ASSERT(nodeOffset <= invalidOffset);
        dstNodes[idx].store(static_cast<CompressedType>(nodeOffset), std::memory_order_relaxed);
    }

    common::offset_t getInvalidOffset() const override { return invalidOffset; }

    std::span<std::atomic<CompressedType>> dstNodes;
    common::offset_t invalidOffset;
};

common::offset_t minNumBytesToStore(common::offset_t value) {
    const auto bitWidth = std::bit_width(value);
    static constexpr decltype(bitWidth) bitsPerByte = 8;
    return std::bit_ceil(static_cast<common::offset_t>(common::ceilDiv(bitWidth, bitsPerByte)));
}
} // namespace

CompressedNodeOffsetBuffer::CompressedNodeOffsetBuffer(MemoryManager* mm, common::offset_t numNodes,
    common::length_t maxDegree) {
    const auto numEntries = numNodes * maxDegree;
    switch (minNumBytesToStore(numNodes)) {
    case 8: {
        buffer = mm->allocateBuffer(false, numEntries * sizeof(std::atomic<uint64_t>));
        view = std::make_unique<TypedCompressedView<uint64_t>>(buffer->getData(), numEntries);
    } break;
    case 4: {
        buffer = mm->allocateBuffer(false, numEntries * sizeof(std::atomic<uint32_t>));
        view = std::make_unique<TypedCompressedView<uint32_t>>(buffer->getData(), numEntries);
    } break;
    case 2: {
        buffer = mm->allocateBuffer(false, numEntries * sizeof(std::atomic<uint16_t>));
        view = std::make_unique<TypedCompressedView<uint16_t>>(buffer->getData(), numEntries);
    } break;
    case 1: {
        buffer = mm->allocateBuffer(false, numEntries * sizeof(std::atomic<uint8_t>));
        view = std::make_unique<TypedCompressedView<uint8_t>>(buffer->getData(), numEntries);
    } break;
    default:
        KU_UNREACHABLE;
    }
}

compressed_offsets_t CompressedNodeOffsetBuffer::getNeighbors(common::offset_t nodeOffset,
    common::offset_t maxDegree, common::offset_t numNbrs) const {
    auto startOffset = nodeOffset * maxDegree;
    return compressed_offsets_t{*view, startOffset, startOffset + numNbrs};
}

InMemHNSWGraph::InMemHNSWGraph(MemoryManager* mm, common::offset_t numNodes,
    common::length_t maxDegree)
    : numNodes{numNodes}, dstNodes(mm, numNodes, maxDegree), maxDegree{maxDegree},
      invalidOffset(dstNodes.getInvalidOffset()) {
    KU_ASSERT(invalidOffset > 0);
    csrLengthBuffer = mm->allocateBuffer(true, numNodes * sizeof(std::atomic<uint16_t>));
    csrLengths = reinterpret_cast<std::atomic<uint16_t>*>(csrLengthBuffer->getData());
    resetCSRLengthAndDstNodes();
}

// NOLINTNEXTLINE(readability-make-member-function-const): Semantically non-const function.
void InMemHNSWGraph::finalize(MemoryManager& mm, common::node_group_idx_t nodeGroupIdx,
    const processor::PartitionerSharedState& partitionerSharedState) {
    const auto& partitionBuffers = partitionerSharedState.partitioningBuffers[0]->partitions;
    auto numRels = 0u;
    const auto startNodeOffset = StorageUtils::getStartOffsetOfNodeGroup(nodeGroupIdx);
    const auto numNodesInGroup =
        std::min(common::StorageConfig::NODE_GROUP_SIZE, numNodes - startNodeOffset);
    for (auto i = 0u; i < numNodesInGroup; i++) {
        numRels += getCSRLength(startNodeOffset + i);
    }
    finalizeNodeGroup(mm, nodeGroupIdx, numRels, partitionerSharedState.srcNodeTable->getTableID(),
        partitionerSharedState.dstNodeTable->getTableID(),
        partitionerSharedState.relTable->getTableID(), *partitionBuffers[nodeGroupIdx]);
}

void InMemHNSWGraph::finalizeNodeGroup(MemoryManager& mm, common::node_group_idx_t nodeGroupIdx,
    uint64_t numRels, common::table_id_t srcNodeTableID, common::table_id_t dstNodeTableID,
    common::table_id_t relTableID, InMemChunkedNodeGroupCollection& partition) const {
    const auto startNodeOffset = StorageUtils::getStartOffsetOfNodeGroup(nodeGroupIdx);
    const auto numNodesInGroup =
        std::min(common::StorageConfig::NODE_GROUP_SIZE, numNodes - startNodeOffset);
    // BOUND_ID, NBR_ID, REL_ID.
    std::vector<common::LogicalType> columnTypes;
    columnTypes.push_back(common::LogicalType::INTERNAL_ID());
    columnTypes.push_back(common::LogicalType::INTERNAL_ID());
    columnTypes.push_back(common::LogicalType::INTERNAL_ID());
    auto chunkedNodeGroup = std::make_unique<ChunkedNodeGroup>(mm, columnTypes,
        false /* enableCompression */, numRels, 0 /* startRowIdx */, ResidencyState::IN_MEMORY);

    auto currNumRels = 0u;
    auto& boundColumnChunk = chunkedNodeGroup->getColumnChunk(0).getData();
    auto& nbrColumnChunk = chunkedNodeGroup->getColumnChunk(1).getData();
    auto& relIDColumnChunk = chunkedNodeGroup->getColumnChunk(2).getData();
    boundColumnChunk.cast<InternalIDChunkData>().setTableID(srcNodeTableID);
    nbrColumnChunk.cast<InternalIDChunkData>().setTableID(dstNodeTableID);
    relIDColumnChunk.cast<InternalIDChunkData>().setTableID(relTableID);
    for (auto i = 0u; i < numNodesInGroup; i++) {
        const auto currNodeOffset = startNodeOffset + i;
        const auto csrLen = getCSRLength(currNodeOffset);
        const auto csrOffset = currNodeOffset * maxDegree;
        for (auto j = 0u; j < csrLen; j++) {
            boundColumnChunk.setValue<common::offset_t>(currNodeOffset, currNumRels);
            relIDColumnChunk.setValue<common::offset_t>(currNumRels, currNumRels);
            const auto nbrOffset = getDstNode(csrOffset + j);
            KU_ASSERT(nbrOffset < numNodes);
            nbrColumnChunk.setValue<common::offset_t>(nbrOffset, currNumRels);
            currNumRels++;
        }
    }
    chunkedNodeGroup->setNumRows(currNumRels);

    for (auto i = 0u; i < nbrColumnChunk.getNumValues(); i++) {
        const auto offset = nbrColumnChunk.getValue<common::offset_t>(i);
        KU_ASSERT(offset < numNodes);
        KU_UNUSED(offset);
    }
    chunkedNodeGroup->setUnused(mm);
    partition.merge(std::move(chunkedNodeGroup));
}

void InMemHNSWGraph::resetCSRLengthAndDstNodes() {
    for (common::offset_t i = 0; i < numNodes; i++) {
        setCSRLength(i, 0);
    }
    for (common::offset_t i = 0; i < numNodes * maxDegree; i++) {
        setDstNode(i, getInvalidOffset());
    }
}

} // namespace vector_extension
} // namespace kuzu

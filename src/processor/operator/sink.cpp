#include "processor/operator/sink.h"
#include "processor/result/factorized_table_util.h"

namespace kuzu {
namespace processor {

void SinkSharedState::mergeLocalTable(FactorizedTable& localTable) {
    std::unique_lock lck{mtx};
    table->merge(localTable);
}

void SinkSharedState::appendString(const std::string& str) {
    KU_ASSERT(table->getTableSchema()->getNumColumns() == 1);
    FactorizedTableUtils::appendStringToTable(table.get(), str, table->getMemoryManager());
}

std::unique_ptr<ResultSet> Sink::getResultSet(storage::MemoryManager* memoryManager) {
    if (resultSetDescriptor == nullptr) {
        // Some pipeline does not need a resultSet, e.g. OrderByMerge
        return std::unique_ptr<ResultSet>();
    }
    return std::make_unique<ResultSet>(resultSetDescriptor.get(), memoryManager);
}

void Sink::execute(ResultSet* resultSet, ExecutionContext* context) {
    initLocalState(resultSet, context);
    metrics->executionTime.start();
    executeInternal(context);
    metrics->executionTime.stop();
}

} // namespace processor
} // namespace kuzu

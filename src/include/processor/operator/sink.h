#pragma once

#include "common/exception/internal.h"
#include "common/metric.h"
#include "processor/operator/physical_operator.h"
#include "processor/result/factorized_table.h"
#include "processor/result/result_set_descriptor.h"

namespace kuzu {
namespace processor {

class SinkSharedState {
public:
    explicit SinkSharedState(std::shared_ptr<FactorizedTable> table)
            : table{std::move(table)} {}

    void mergeLocalTable(FactorizedTable& localTable);

    void appendString(const std::string& str);

    std::shared_ptr<FactorizedTable> getTable() { return table; }

private:
    std::mutex mtx;
    std::shared_ptr<FactorizedTable> table;
};

class KUZU_API Sink : public PhysicalOperator {
public:
    // Leaf sink operator.
    Sink(PhysicalOperatorType operatorType, physical_op_id id,
        std::unique_ptr<OPPrintInfo> printInfo)
        : PhysicalOperator{operatorType, id, std::move(printInfo)} {}
    // Unary sink operator.
    Sink(PhysicalOperatorType operatorType, std::unique_ptr<PhysicalOperator> child,
        physical_op_id id, std::unique_ptr<OPPrintInfo> printInfo)
        : PhysicalOperator{operatorType, std::move(child), id, std::move(printInfo)} {}
    // Unary sink operator with result table.
    Sink(PhysicalOperatorType operatorType, std::shared_ptr<SinkSharedState> sinkSharedState, std::unique_ptr<PhysicalOperator> child,
        physical_op_id id, std::unique_ptr<OPPrintInfo> printInfo)
        : PhysicalOperator{operatorType, std::move(child), id, std::move(printInfo)}, sinkSharedState{std::move(sinkSharedState)} {}

    bool isSink() const override { return true; }

    void setDescriptor(std::unique_ptr<ResultSetDescriptor> descriptor) {
        KU_ASSERT(resultSetDescriptor == nullptr);
        resultSetDescriptor = std::move(descriptor);
    }
    std::unique_ptr<ResultSet> getResultSet(storage::MemoryManager* memoryManager);

    bool hasResultTable() const {
        return sinkSharedState != nullptr;
    }
    std::shared_ptr<FactorizedTable> getResultTable() const {
        KU_ASSERT(hasResultTable);
        return sinkSharedState->getTable();
    }

    void execute(ResultSet* resultSet, ExecutionContext* context);

    std::unique_ptr<PhysicalOperator> copy() override = 0;

protected:
    virtual void executeInternal(ExecutionContext* context) = 0;

    bool getNextTuplesInternal(ExecutionContext* /*context*/) final {
        throw common::InternalException(
            "getNextTupleInternal() should not be called on sink operator.");
    }

protected:
    std::unique_ptr<ResultSetDescriptor> resultSetDescriptor;
    std::shared_ptr<SinkSharedState> sinkSharedState;
};

class KUZU_API DummySink final : public Sink {
    static constexpr PhysicalOperatorType type_ = PhysicalOperatorType::DUMMY_SINK;

public:
    DummySink(std::unique_ptr<PhysicalOperator> child, uint32_t id)
        : Sink{type_, std::move(child), id, OPPrintInfo::EmptyInfo()} {}

    std::unique_ptr<PhysicalOperator> copy() override {
        return std::make_unique<DummySink>(children[0]->copy(), id);
    }

protected:
    void executeInternal(ExecutionContext* context) override {
        while (children[0]->getNextTuple(context)) {
            // DO NOTHING.
        }
    }
};

} // namespace processor
} // namespace kuzu

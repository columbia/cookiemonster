from typing import Dict, List, Union, Tuple
from systemx.budget import BasicBudget

kOk = "OK"
kInsufficientBudgetError = "InsufficientBudgetError"


class BudgetAccountantKey:
    def __init__(self, block):
        self.key = f"{block}"


class BudgetAccountantResult:
    def __init__(self, status: str, total_budget_consumed: float) -> None:
        self.status = status
        self.total_budget_consumed = total_budget_consumed

    def succeeded(self):
        return self.status == kOk


class BudgetAccountant:
    def __init__(self) -> None:
        self.filter: Dict[str, float] = {}

    def get_blocks_count(self):
        return len(self.filter.keys())

    def update_block_budget(self, block, budget):
        key = BudgetAccountantKey(block).key
        # Add budget in the key value store
        self.filter[key] = budget

    def get_block_budget(self, block):
        """Returns the remaining budget of block"""
        key = BudgetAccountantKey(block).key
        if key in self.filter:
            return self.filter[key]
        # logger.info(f"Block {block} does not exist")
        return None

    def get_all_block_budgets(self):
        return {block: budget.epsilon for block, budget in self.filter.items()}

    def add_new_block_budget(self, block, initial_budget):
        assert block not in self.filter
        budget = BasicBudget(initial_budget)
        self.update_block_budget(block, budget)

    def can_run(self, blocks: Union[int, List[int], Tuple[int, int]], run_budget):
        if isinstance(blocks, int):
            blocks = [blocks]
        elif isinstance(blocks, tuple):
            blocks = range(blocks[0], blocks[1] + 1)

        for block in blocks:
            budget = self.get_block_budget(block)
            if not budget.can_allocate(run_budget):
                return False
        return True

    def consume_block_budget(self, block, run_budget):
        """Consumes 'run_budget' from the remaining block budget"""
        budget = self.get_block_budget(block)
        budget -= run_budget
        # Re-write the budget in the KV store
        self.update_block_budget(block, budget)

    def pay_all_or_nothing(
        self, blocks: Union[Tuple[int, int], List[int]], epsilon: float
    ) -> BudgetAccountantResult:
        if isinstance(blocks, tuple):
            blocks = block_window_to_list(blocks)

        total_budget_consumed = 0

        # Check if all blocks have enough remaining budget
        for block in blocks:
            # Check if epoch has enough budget
            if not self.can_run(block, BasicBudget(epsilon)):
                return BudgetAccountantResult(
                    kInsufficientBudgetError, total_budget_consumed
                )

        # Consume budget from all blocks
        for block in blocks:
            self.consume_block_budget(block, BasicBudget(epsilon))
            total_budget_consumed += epsilon

        return BudgetAccountantResult(kOk, total_budget_consumed)

    def maybe_initialize_filter(
        self, blocks: Union[Tuple[int, int], List[int]], initial_budget: float
    ):
        if isinstance(blocks, tuple):
            blocks = block_window_to_list(blocks)

        for block in blocks:
            # Maybe initialize epoch
            if self.get_block_budget(block) is None:
                self.add_new_block_budget(block, initial_budget)

    def dump(self):
        budgets = [
            (block, budget.dump()) for (block, budget) in self.get_all_block_budgets()
        ]
        return budgets


def block_window_to_list(blocks: Tuple[int, int]) -> List[int]:
    return list(range(blocks[0], blocks[1] + 1))

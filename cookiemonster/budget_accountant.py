from typing import Dict, List, Union, Tuple

kOk = "OK"
kInsufficientBudgetError = "InsufficientBudgetError"


class BudgetAccountantResult:
    def __init__(self, status: str) -> None:
        self.status = status

    def succeeded(self):
        return self.status == kOk


class BudgetAccountant:
    def __init__(self, initial_budget) -> None:
        self.initial_budget = initial_budget
        self.filter: Dict[str, float] = {}

    def can_run(self, blocks: Union[int, List[int], Tuple[int, int]], run_budget):
        if isinstance(blocks, int):
            blocks = [blocks]
        elif isinstance(blocks, tuple):
            blocks = block_window_range(blocks)

        for block in blocks:
            remaining_budget = self.filter[block]
            if remaining_budget < run_budget:
                return False
        return True

    def pay_all_or_nothing(
        self,
        blocks: Union[Tuple[int, int], List[int]],
        epsilon: float,
    ) -> BudgetAccountantResult:

        if isinstance(blocks, tuple):
            blocks = block_window_range(blocks)

        # Check if all blocks have enough remaining budget
        for block in blocks:
            if not self.can_run(block, epsilon):
                return BudgetAccountantResult(kInsufficientBudgetError)

        # Consume budget from all blocks
        for block in blocks:
            self.filter[block] -= epsilon

        return BudgetAccountantResult(kOk)

    def maybe_initialize_filter(self, blocks: Union[Tuple[int, int], List[int]]):
        if isinstance(blocks, tuple):
            blocks = block_window_range(blocks)

        for block in blocks:
            if block not in self.filter:
                self.filter[block] = self.initial_budget

    def get_max_consumption_across_blocks(self):
        return self.initial_budget - min(self.filter.values())

    def get_sum_consumption_across_blocks(self):
        remaining_budgets = self.filter.values()
        sum_initial_budgets = len(remaining_budgets) * self.initial_budget
        return sum_initial_budgets - sum(remaining_budgets)

    def get_consumption_per_block(self):
        remaining_budgets = self.filter.values()
        consumption_per_block = [
            self.initial_budget - remaining_budget
            for remaining_budget in remaining_budgets
        ]
        return consumption_per_block


def block_window_range(blocks: Tuple[int, int]):
    return range(blocks[0], blocks[1] + 1)

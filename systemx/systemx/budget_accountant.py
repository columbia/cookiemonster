from typing import Dict, List, Union, Tuple
from systemx.budget import BasicBudget


class BudgetAccountantKey:
    def __init__(self, block):
        self.key = f"{block}"


class BudgetAccountant:
    def __init__(self, config) -> None:
        self.config = config
        self.epsilon = float(self.config.initial_budget)
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
        return self.filter.items()

    def add_new_block_budget(self, block):
        assert block not in self.filter
        budget = BasicBudget(self.epsilon)
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

    def dump(self):
        budgets = [
            (block, budget.dump()) for (block, budget) in self.get_all_block_budgets()
        ]
        return budgets

"""Simple account-based balance system for PoML simulation."""

from __future__ import annotations

import logging
import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from poml_sim.types import Transaction

logger = logging.getLogger(__name__)


class AccountState:
    """Account balances mapped by public key."""

    def __init__(self) -> None:
        self.balances: dict[bytes, int] = {}

    def get_balance(self, pk: bytes) -> int:
        return self.balances.get(pk, 0)

    def credit(self, pk: bytes, amount: int) -> None:
        self.balances[pk] = self.get_balance(pk) + amount

    def debit(self, pk: bytes, amount: int) -> bool:
        bal = self.get_balance(pk)
        if bal < amount:
            return False
        self.balances[pk] = bal - amount
        return True

    def apply_block_reward(self, miner_pk: bytes, reward: int) -> None:
        self.credit(miner_pk, reward)
        logger.debug("Block reward %d → %s", reward, miner_pk.hex()[:12])

    def apply_transaction(self, tx: Transaction) -> bool:
        if not self.validate_transaction(tx):
            return False
        self.debit(tx.sender, tx.amount)
        self.credit(tx.receiver, tx.amount)
        return True

    def validate_transaction(self, tx: Transaction) -> bool:
        if tx.amount <= 0:
            return False
        if self.get_balance(tx.sender) < tx.amount:
            return False
        return True

    def apply_fees(self, miner_pk: bytes, fees: int) -> None:
        """Credit accumulated query fees to the miner."""
        if fees > 0:
            self.credit(miner_pk, fees)

    def copy(self) -> AccountState:
        new = AccountState()
        new.balances = dict(self.balances)
        return new

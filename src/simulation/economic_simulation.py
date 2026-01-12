"""
Economic Simulation Module

Advanced economic simulation including:
- Market Dynamics (supply/demand, price discovery)
- Agent-Based Economic Modeling
- Auction Mechanisms
- Trading and Exchange
- Credit and Banking
- Network Effects
- Market Microstructure
- Behavioral Economics

Based on:
- Agent-Based Computational Economics
- Market Microstructure Theory
- Auction Theory
- Behavioral Finance
"""

import random
import math
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import heapq


# ============================================================================
# BASIC ECONOMIC TYPES
# ============================================================================

class GoodType(Enum):
    """Types of economic goods"""
    COMMODITY = "commodity"  # Fungible physical goods
    SERVICE = "service"
    ASSET = "asset"  # Stocks, bonds, etc.
    CURRENCY = "currency"
    INFORMATION = "information"
    LABOR = "labor"


class OrderType(Enum):
    """Types of market orders"""
    MARKET = "market"  # Execute immediately at best price
    LIMIT = "limit"  # Execute at specified price or better
    STOP = "stop"  # Trigger when price reaches level
    ICEBERG = "iceberg"  # Hidden quantity


class OrderSide(Enum):
    """Buy or sell"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Good:
    """An economic good"""
    id: str
    name: str
    good_type: GoodType
    base_value: float = 1.0
    divisible: bool = True
    perishable: bool = False
    decay_rate: float = 0.0


@dataclass
class Order:
    """A market order"""
    id: str
    agent_id: str
    good_id: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # For limit orders
    timestamp: datetime = field(default_factory=datetime.now)
    filled: float = 0.0
    active: bool = True

    @property
    def remaining(self) -> float:
        return self.quantity - self.filled


@dataclass
class Trade:
    """A completed trade"""
    id: str
    good_id: str
    buyer_id: str
    seller_id: str
    quantity: float
    price: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Portfolio:
    """Agent's portfolio of goods and currency"""
    agent_id: str
    cash: float = 1000.0
    holdings: Dict[str, float] = field(default_factory=dict)  # Good -> quantity
    liabilities: Dict[str, float] = field(default_factory=dict)

    def get_holding(self, good_id: str) -> float:
        return self.holdings.get(good_id, 0.0)

    def add_holding(self, good_id: str, quantity: float):
        self.holdings[good_id] = self.holdings.get(good_id, 0) + quantity

    def remove_holding(self, good_id: str, quantity: float) -> bool:
        if self.holdings.get(good_id, 0) >= quantity:
            self.holdings[good_id] -= quantity
            return True
        return False


# ============================================================================
# MARKET MECHANISMS
# ============================================================================

class OrderBook:
    """Order book for a single good"""

    def __init__(self, good_id: str):
        self.good_id = good_id
        self.bids: List[Tuple[float, Order]] = []  # Max heap by price (negative for max)
        self.asks: List[Tuple[float, Order]] = []  # Min heap by price
        self.trades: List[Trade] = []
        self.last_price: Optional[float] = None

    def add_order(self, order: Order) -> List[Trade]:
        """Add an order and execute any matches"""
        if order.good_id != self.good_id:
            return []

        trades = []

        if order.order_type == OrderType.MARKET:
            trades = self._execute_market_order(order)
        elif order.order_type == OrderType.LIMIT:
            trades = self._execute_limit_order(order)

        return trades

    def _execute_market_order(self, order: Order) -> List[Trade]:
        """Execute market order against book"""
        trades = []

        if order.side == OrderSide.BUY:
            # Match against asks
            while order.remaining > 0 and self.asks:
                best_ask_price, best_ask = heapq.heappop(self.asks)

                trade = self._match_orders(order, best_ask, best_ask.price)
                if trade:
                    trades.append(trade)

                if best_ask.remaining > 0:
                    heapq.heappush(self.asks, (best_ask.price, best_ask))
        else:
            # Match against bids
            while order.remaining > 0 and self.bids:
                neg_price, best_bid = heapq.heappop(self.bids)

                trade = self._match_orders(best_bid, order, best_bid.price)
                if trade:
                    trades.append(trade)

                if best_bid.remaining > 0:
                    heapq.heappush(self.bids, (-best_bid.price, best_bid))

        return trades

    def _execute_limit_order(self, order: Order) -> List[Trade]:
        """Execute limit order - match if possible, otherwise add to book"""
        trades = []

        if order.price is None:
            return trades

        if order.side == OrderSide.BUY:
            # Try to match against asks at or below limit price
            while order.remaining > 0 and self.asks:
                best_ask_price, best_ask = self.asks[0]
                if best_ask_price > order.price:
                    break

                heapq.heappop(self.asks)
                trade = self._match_orders(order, best_ask, best_ask.price)
                if trade:
                    trades.append(trade)

                if best_ask.remaining > 0:
                    heapq.heappush(self.asks, (best_ask.price, best_ask))

            # Add remaining to book
            if order.remaining > 0:
                heapq.heappush(self.bids, (-order.price, order))

        else:
            # Try to match against bids at or above limit price
            while order.remaining > 0 and self.bids:
                neg_price, best_bid = self.bids[0]
                best_bid_price = -neg_price
                if best_bid_price < order.price:
                    break

                heapq.heappop(self.bids)
                trade = self._match_orders(best_bid, order, best_bid_price)
                if trade:
                    trades.append(trade)

                if best_bid.remaining > 0:
                    heapq.heappush(self.bids, (-best_bid.price, best_bid))

            # Add remaining to book
            if order.remaining > 0:
                heapq.heappush(self.asks, (order.price, order))

        return trades

    def _match_orders(self, buy_order: Order, sell_order: Order,
                     price: float) -> Optional[Trade]:
        """Match two orders at a price"""
        quantity = min(buy_order.remaining, sell_order.remaining)

        if quantity <= 0:
            return None

        buy_order.filled += quantity
        sell_order.filled += quantity

        trade = Trade(
            id=f"trade_{len(self.trades)}",
            good_id=self.good_id,
            buyer_id=buy_order.agent_id,
            seller_id=sell_order.agent_id,
            quantity=quantity,
            price=price
        )

        self.trades.append(trade)
        self.last_price = price

        return trade

    def get_best_bid(self) -> Optional[float]:
        """Get best bid price"""
        if self.bids:
            return -self.bids[0][0]
        return None

    def get_best_ask(self) -> Optional[float]:
        """Get best ask price"""
        if self.asks:
            return self.asks[0][0]
        return None

    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread"""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid and ask:
            return ask - bid
        return None

    def get_midpoint(self) -> Optional[float]:
        """Get midpoint price"""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid and ask:
            return (bid + ask) / 2
        return None


class Market:
    """A complete market with multiple goods"""

    def __init__(self, name: str):
        self.name = name
        self.goods: Dict[str, Good] = {}
        self.order_books: Dict[str, OrderBook] = {}
        self.agents: Dict[str, Portfolio] = {}
        self.all_trades: List[Trade] = []
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)

    def register_good(self, good: Good):
        """Register a good for trading"""
        self.goods[good.id] = good
        self.order_books[good.id] = OrderBook(good.id)

    def register_agent(self, portfolio: Portfolio):
        """Register an agent"""
        self.agents[portfolio.agent_id] = portfolio

    def submit_order(self, order: Order) -> List[Trade]:
        """Submit an order to the market"""
        if order.good_id not in self.order_books:
            return []

        if order.agent_id not in self.agents:
            return []

        # Validate order
        if not self._validate_order(order):
            return []

        # Execute
        trades = self.order_books[order.good_id].add_order(order)

        # Settle trades
        for trade in trades:
            self._settle_trade(trade)
            self.all_trades.append(trade)
            self.price_history[trade.good_id].append(
                (trade.timestamp, trade.price)
            )

        return trades

    def _validate_order(self, order: Order) -> bool:
        """Validate an order"""
        portfolio = self.agents.get(order.agent_id)
        if not portfolio:
            return False

        if order.side == OrderSide.BUY:
            # Check cash
            if order.order_type == OrderType.LIMIT and order.price:
                required = order.quantity * order.price
            else:
                # Market order - estimate
                ask = self.order_books[order.good_id].get_best_ask()
                required = order.quantity * (ask or 100)

            return portfolio.cash >= required

        else:
            # Check holdings
            return portfolio.get_holding(order.good_id) >= order.quantity

    def _settle_trade(self, trade: Trade):
        """Settle a completed trade"""
        buyer = self.agents.get(trade.buyer_id)
        seller = self.agents.get(trade.seller_id)

        if buyer and seller:
            total_cost = trade.quantity * trade.price

            buyer.cash -= total_cost
            buyer.add_holding(trade.good_id, trade.quantity)

            seller.cash += total_cost
            seller.remove_holding(trade.good_id, trade.quantity)

    def get_price(self, good_id: str) -> Optional[float]:
        """Get current price of a good"""
        book = self.order_books.get(good_id)
        if book:
            return book.last_price or book.get_midpoint()
        return None

    def get_market_summary(self) -> Dict[str, Any]:
        """Get market summary"""
        return {
            "name": self.name,
            "goods": len(self.goods),
            "agents": len(self.agents),
            "total_trades": len(self.all_trades),
            "prices": {
                good_id: self.get_price(good_id)
                for good_id in self.goods
            }
        }


# ============================================================================
# AUCTION MECHANISMS
# ============================================================================

class AuctionType(Enum):
    """Types of auctions"""
    ENGLISH = "english"  # Ascending price
    DUTCH = "dutch"  # Descending price
    FIRST_PRICE_SEALED = "first_price"
    SECOND_PRICE_SEALED = "second_price"  # Vickrey
    ALL_PAY = "all_pay"


@dataclass
class Bid:
    """A bid in an auction"""
    bidder_id: str
    amount: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AuctionResult:
    """Result of an auction"""
    winner_id: Optional[str]
    winning_bid: Optional[float]
    payment: float
    all_bids: List[Bid]


class Auction:
    """Generic auction mechanism"""

    def __init__(self, item_id: str, auction_type: AuctionType,
                 reserve_price: float = 0.0):
        self.item_id = item_id
        self.auction_type = auction_type
        self.reserve_price = reserve_price
        self.bids: List[Bid] = []
        self.is_open: bool = True
        self.result: Optional[AuctionResult] = None

        # For English auctions
        self.current_price = reserve_price
        self.increment = 1.0

        # For Dutch auctions
        self.starting_price = 100.0
        self.decrement = 1.0

    def submit_bid(self, bidder_id: str, amount: float) -> bool:
        """Submit a bid"""
        if not self.is_open:
            return False

        bid = Bid(bidder_id=bidder_id, amount=amount)
        self.bids.append(bid)

        if self.auction_type == AuctionType.ENGLISH:
            if amount > self.current_price:
                self.current_price = amount
                return True
            return False

        elif self.auction_type == AuctionType.DUTCH:
            # First bidder at current price wins
            if amount >= self.current_price:
                self._close_auction()
                return True
            return False

        return True

    def dutch_tick(self) -> bool:
        """Decrease Dutch auction price"""
        if self.auction_type != AuctionType.DUTCH:
            return False

        self.current_price -= self.decrement
        if self.current_price <= self.reserve_price:
            self._close_auction()
            return False
        return True

    def close(self) -> AuctionResult:
        """Close the auction and determine winner"""
        self._close_auction()
        return self.result

    def _close_auction(self):
        """Internal close logic"""
        self.is_open = False

        if not self.bids:
            self.result = AuctionResult(
                winner_id=None,
                winning_bid=None,
                payment=0,
                all_bids=self.bids
            )
            return

        if self.auction_type == AuctionType.ENGLISH:
            # Highest bidder wins, pays their bid
            sorted_bids = sorted(self.bids, key=lambda b: b.amount, reverse=True)
            winner = sorted_bids[0]

            if winner.amount >= self.reserve_price:
                self.result = AuctionResult(
                    winner_id=winner.bidder_id,
                    winning_bid=winner.amount,
                    payment=winner.amount,
                    all_bids=self.bids
                )
            else:
                self.result = AuctionResult(
                    winner_id=None,
                    winning_bid=None,
                    payment=0,
                    all_bids=self.bids
                )

        elif self.auction_type == AuctionType.DUTCH:
            # First bidder wins, pays current price
            winner = self.bids[-1] if self.bids else None
            if winner:
                self.result = AuctionResult(
                    winner_id=winner.bidder_id,
                    winning_bid=winner.amount,
                    payment=self.current_price,
                    all_bids=self.bids
                )
            else:
                self.result = AuctionResult(
                    winner_id=None,
                    winning_bid=None,
                    payment=0,
                    all_bids=self.bids
                )

        elif self.auction_type == AuctionType.FIRST_PRICE_SEALED:
            sorted_bids = sorted(self.bids, key=lambda b: b.amount, reverse=True)
            winner = sorted_bids[0]

            if winner.amount >= self.reserve_price:
                self.result = AuctionResult(
                    winner_id=winner.bidder_id,
                    winning_bid=winner.amount,
                    payment=winner.amount,  # Pay own bid
                    all_bids=self.bids
                )
            else:
                self.result = AuctionResult(
                    winner_id=None,
                    winning_bid=None,
                    payment=0,
                    all_bids=self.bids
                )

        elif self.auction_type == AuctionType.SECOND_PRICE_SEALED:
            sorted_bids = sorted(self.bids, key=lambda b: b.amount, reverse=True)
            winner = sorted_bids[0]
            second = sorted_bids[1] if len(sorted_bids) > 1 else winner

            if winner.amount >= self.reserve_price:
                self.result = AuctionResult(
                    winner_id=winner.bidder_id,
                    winning_bid=winner.amount,
                    payment=max(second.amount, self.reserve_price),  # Pay second price
                    all_bids=self.bids
                )
            else:
                self.result = AuctionResult(
                    winner_id=None,
                    winning_bid=None,
                    payment=0,
                    all_bids=self.bids
                )


# ============================================================================
# ECONOMIC AGENTS
# ============================================================================

class AgentStrategy(Enum):
    """Trading strategies"""
    RANDOM = "random"
    FUNDAMENTAL = "fundamental"  # Trade based on fundamental value
    MOMENTUM = "momentum"  # Follow trends
    MEAN_REVERSION = "mean_reversion"  # Bet on return to mean
    MARKET_MAKER = "market_maker"  # Provide liquidity


@dataclass
class EconomicAgent:
    """An economic agent with trading behavior"""
    id: str
    portfolio: Portfolio
    strategy: AgentStrategy = AgentStrategy.RANDOM
    risk_tolerance: float = 0.5
    patience: float = 0.5  # How long willing to wait

    # Beliefs
    value_beliefs: Dict[str, float] = field(default_factory=dict)  # Good -> believed value

    # Performance
    trades_made: int = 0
    profit_loss: float = 0.0


class TradingAgent:
    """Agent that can trade in markets"""

    def __init__(self, agent: EconomicAgent, market: Market):
        self.agent = agent
        self.market = market
        self.order_history: List[Order] = []

    def generate_order(self, good_id: str) -> Optional[Order]:
        """Generate an order based on strategy"""
        if good_id not in self.market.goods:
            return None

        current_price = self.market.get_price(good_id)
        if not current_price:
            current_price = self.market.goods[good_id].base_value

        believed_value = self.agent.value_beliefs.get(
            good_id,
            self.market.goods[good_id].base_value
        )

        if self.agent.strategy == AgentStrategy.RANDOM:
            return self._random_order(good_id, current_price)
        elif self.agent.strategy == AgentStrategy.FUNDAMENTAL:
            return self._fundamental_order(good_id, current_price, believed_value)
        elif self.agent.strategy == AgentStrategy.MOMENTUM:
            return self._momentum_order(good_id)
        elif self.agent.strategy == AgentStrategy.MEAN_REVERSION:
            return self._mean_reversion_order(good_id, current_price)
        elif self.agent.strategy == AgentStrategy.MARKET_MAKER:
            return self._market_maker_order(good_id, current_price)

        return None

    def _random_order(self, good_id: str, current_price: float) -> Order:
        """Generate random order"""
        side = random.choice([OrderSide.BUY, OrderSide.SELL])
        quantity = random.uniform(1, 10)
        price = current_price * random.uniform(0.95, 1.05)

        return Order(
            id=f"order_{len(self.order_history)}",
            agent_id=self.agent.id,
            good_id=good_id,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price
        )

    def _fundamental_order(self, good_id: str, current_price: float,
                          believed_value: float) -> Optional[Order]:
        """Trade based on fundamental value"""
        # Buy if undervalued, sell if overvalued
        value_diff = believed_value - current_price

        if abs(value_diff) < current_price * 0.05:
            return None  # Not enough mispricing

        if value_diff > 0:
            # Undervalued - buy
            side = OrderSide.BUY
            price = current_price * 1.01
        else:
            # Overvalued - sell
            side = OrderSide.SELL
            price = current_price * 0.99

        quantity = min(10, abs(value_diff))

        return Order(
            id=f"order_{len(self.order_history)}",
            agent_id=self.agent.id,
            good_id=good_id,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price
        )

    def _momentum_order(self, good_id: str) -> Optional[Order]:
        """Trade based on price momentum"""
        history = self.market.price_history.get(good_id, [])

        if len(history) < 5:
            return None

        recent_prices = [p for _, p in history[-5:]]
        momentum = recent_prices[-1] - recent_prices[0]

        if abs(momentum) < 0.5:
            return None

        side = OrderSide.BUY if momentum > 0 else OrderSide.SELL
        current = recent_prices[-1]
        price = current * (1.02 if side == OrderSide.BUY else 0.98)

        return Order(
            id=f"order_{len(self.order_history)}",
            agent_id=self.agent.id,
            good_id=good_id,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=5,
            price=price
        )

    def _mean_reversion_order(self, good_id: str,
                             current_price: float) -> Optional[Order]:
        """Trade expecting mean reversion"""
        history = self.market.price_history.get(good_id, [])

        if len(history) < 20:
            return None

        recent_prices = [p for _, p in history[-20:]]
        mean = sum(recent_prices) / len(recent_prices)

        deviation = (current_price - mean) / mean

        if abs(deviation) < 0.1:
            return None

        # Trade against deviation
        side = OrderSide.SELL if deviation > 0 else OrderSide.BUY
        price = current_price * (0.99 if side == OrderSide.SELL else 1.01)

        return Order(
            id=f"order_{len(self.order_history)}",
            agent_id=self.agent.id,
            good_id=good_id,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=3,
            price=price
        )

    def _market_maker_order(self, good_id: str,
                           current_price: float) -> Order:
        """Provide liquidity by placing orders on both sides"""
        spread = 0.02
        bid = current_price * (1 - spread)
        ask = current_price * (1 + spread)

        # Alternate between bid and ask
        if random.random() < 0.5:
            return Order(
                id=f"order_{len(self.order_history)}",
                agent_id=self.agent.id,
                good_id=good_id,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=10,
                price=bid
            )
        else:
            return Order(
                id=f"order_{len(self.order_history)}",
                agent_id=self.agent.id,
                good_id=good_id,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=10,
                price=ask
            )


# ============================================================================
# ECONOMIC SIMULATION
# ============================================================================

class EconomicSimulation:
    """
    Complete economic simulation with multiple agents and markets.
    """

    def __init__(self, name: str = "Economy"):
        self.name = name
        self.market = Market(name)
        self.agents: Dict[str, EconomicAgent] = {}
        self.trading_agents: Dict[str, TradingAgent] = {}

        self.tick_count = 0
        self.gdp_history: List[float] = []
        self.inflation_history: List[float] = []

    def add_good(self, good: Good):
        """Add a good to the economy"""
        self.market.register_good(good)

    def add_agent(self, agent: EconomicAgent):
        """Add an agent to the economy"""
        self.agents[agent.id] = agent
        self.market.register_agent(agent.portfolio)
        self.trading_agents[agent.id] = TradingAgent(agent, self.market)

    def step(self) -> Dict[str, Any]:
        """Run one simulation step"""
        self.tick_count += 1
        step_trades = []

        # Each agent may trade
        for agent_id, trading_agent in self.trading_agents.items():
            for good_id in self.market.goods:
                if random.random() < 0.3:  # 30% chance to trade each good
                    order = trading_agent.generate_order(good_id)
                    if order:
                        trades = self.market.submit_order(order)
                        step_trades.extend(trades)
                        trading_agent.order_history.append(order)

        # Calculate metrics
        total_value = sum(
            portfolio.cash + sum(
                portfolio.holdings.get(g, 0) * (self.market.get_price(g) or 0)
                for g in self.market.goods
            )
            for portfolio in self.market.agents.values()
        )

        self.gdp_history.append(total_value)

        # Calculate inflation from price changes
        if len(self.market.all_trades) > 10:
            old_prices = [t.price for t in self.market.all_trades[-20:-10]]
            new_prices = [t.price for t in self.market.all_trades[-10:]]
            if old_prices and new_prices:
                inflation = (sum(new_prices) / sum(old_prices) - 1) * 100
                self.inflation_history.append(inflation)

        return {
            "tick": self.tick_count,
            "trades": len(step_trades),
            "total_volume": sum(t.quantity * t.price for t in step_trades),
            "gdp": total_value,
            "market_summary": self.market.get_market_summary()
        }

    def run(self, ticks: int = 100) -> List[Dict[str, Any]]:
        """Run simulation for multiple ticks"""
        results = []
        for _ in range(ticks):
            result = self.step()
            results.append(result)
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get simulation statistics"""
        return {
            "total_ticks": self.tick_count,
            "total_trades": len(self.market.all_trades),
            "total_volume": sum(t.quantity * t.price for t in self.market.all_trades),
            "average_gdp": sum(self.gdp_history) / len(self.gdp_history) if self.gdp_history else 0,
            "average_inflation": sum(self.inflation_history) / len(self.inflation_history) if self.inflation_history else 0,
            "agent_stats": {
                agent_id: {
                    "cash": agent.portfolio.cash,
                    "holdings_value": sum(
                        agent.portfolio.holdings.get(g, 0) * (self.market.get_price(g) or 0)
                        for g in self.market.goods
                    )
                }
                for agent_id, agent in self.agents.items()
            }
        }


# Convenience functions
def create_economic_simulation(name: str = "Economy") -> EconomicSimulation:
    """Create an economic simulation"""
    return EconomicSimulation(name)


def demo_economic_simulation():
    """Demonstrate economic simulation"""
    print("=== Economic Simulation Demo ===")

    sim = create_economic_simulation("Demo Economy")

    # Add goods
    sim.add_good(Good(id="stock_a", name="Stock A", good_type=GoodType.ASSET, base_value=100))
    sim.add_good(Good(id="commodity_b", name="Commodity B", good_type=GoodType.COMMODITY, base_value=50))

    # Add agents with different strategies
    strategies = [
        AgentStrategy.FUNDAMENTAL,
        AgentStrategy.MOMENTUM,
        AgentStrategy.MEAN_REVERSION,
        AgentStrategy.MARKET_MAKER,
        AgentStrategy.RANDOM
    ]

    for i, strategy in enumerate(strategies):
        portfolio = Portfolio(
            agent_id=f"agent_{i}",
            cash=10000,
            holdings={"stock_a": 100, "commodity_b": 200}
        )
        agent = EconomicAgent(
            id=f"agent_{i}",
            portfolio=portfolio,
            strategy=strategy,
            value_beliefs={"stock_a": 100 + random.gauss(0, 10), "commodity_b": 50 + random.gauss(0, 5)}
        )
        sim.add_agent(agent)

    # Run simulation
    print("\nRunning simulation...")
    results = sim.run(ticks=50)

    # Print results
    print(f"\nSimulation complete!")
    print(f"Total ticks: {sim.tick_count}")
    print(f"Total trades: {len(sim.market.all_trades)}")

    stats = sim.get_statistics()
    print(f"Average GDP: {stats['average_gdp']:.2f}")

    print("\nFinal prices:")
    for good_id in sim.market.goods:
        price = sim.market.get_price(good_id)
        print(f"  {good_id}: {price:.2f}" if price else f"  {good_id}: N/A")

    print("\nAgent performance:")
    for agent_id, agent_stats in stats['agent_stats'].items():
        total = agent_stats['cash'] + agent_stats['holdings_value']
        print(f"  {agent_id}: Cash={agent_stats['cash']:.2f}, Holdings={agent_stats['holdings_value']:.2f}, Total={total:.2f}")

    return sim


if __name__ == "__main__":
    demo_economic_simulation()

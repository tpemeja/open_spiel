# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
"""Belote implemented in Python.

Classic (non-contract) French Belote for 4 players in 2 fixed partnerships
(players 0 & 2 vs players 1 & 3). Trump is chosen via the "prise" procedure:
5 cards are dealt to each player, the next card of the stock is turned face
up, and players in turn may take it (round 1) or, if everyone passes, choose
one of the three other suits (round 2). If everyone passes twice, the deal is
redealt with the next player as dealer. This can in principle repeat
indefinitely, so the number of redeals is capped by the "max_redeals" game
parameter (default 10); if every deal keeps failing to produce a taker all
the way up to that cap, the game ends there as a flat draw (all returns 0)
rather than redealing forever. Card play follows standard suit- and
trump-following obligations, and scoring uses the standard 162-point deck
(152 card points + 10 for the last trick), with an all-or-nothing rule: the
declaring team keeps its trick points only if it scores strictly more than the
defenders; otherwise the defending team collects all trick points. If one team
wins all 8 tricks ("capot"), the last-trick bonus is 100 instead of 10, so the deck
is worth 252 points instead of 162, and that full total goes to whichever
team scores higher (the capot-winning team on success, or the defenders'
252 on a failed contract).

The "belote/rebelote" bonus (20 extra points awarded to whichever team has a
single player holding both the King and Queen of the trump suit) is optional
and controlled via the "use_belote_rebelote" game parameter (off by default).
When enabled, the bonus counts toward the declaring team's contract
threshold (its own or the defenders', per official rules) and is always
credited to the holder's team, win or lose.

Official rules (Fédération Française de Belote):
https://www.ffbelote.org/wp-content/uploads/2016/01/regles-officielles-de-la-Belote-27-01-2016.pdf
"""

import numpy as np

import pyspiel

_NUM_PLAYERS = 4
_NUM_SUITS = 4
_NUM_RANKS = 8
_NUM_CARDS = _NUM_SUITS * _NUM_RANKS
_MAX_SCORE = 162
_LAST_TRICK_BONUS = 10
_CAPOT_LAST_TRICK_BONUS = 100
_MAX_SCORE_CAPOT = _MAX_SCORE - _LAST_TRICK_BONUS + _CAPOT_LAST_TRICK_BONUS
_BELOTE_REBELOTE_BONUS = 20
# Safety valve: real belote redeals with no limit if bidding keeps failing,
# but that can never terminate in principle, so redeals are capped and the
# deal ends as a flat draw if the cap is ever exceeded.
_DEFAULT_MAX_REDEALS = 10

_SUIT_NAMES = ["C", "D", "H", "S"]
_RANK_NAMES = ["7", "8", "9", "10", "J", "Q", "K", "A"]

# Card strength, low to high, when the card's suit is NOT trump.
_NONTRUMP_ORDER = ["7", "8", "9", "J", "Q", "K", "10", "A"]
# Card strength, low to high, when the card's suit IS trump.
_TRUMP_ORDER = ["7", "8", "Q", "K", "10", "A", "9", "J"]

_NONTRUMP_POINTS = {
    "7": 0,
    "8": 0,
    "9": 0,
    "J": 2,
    "Q": 3,
    "K": 4,
    "10": 10,
    "A": 11,
}
_TRUMP_POINTS = {
    "7": 0,
    "8": 0,
    "9": 14,
    "J": 20,
    "Q": 3,
    "K": 4,
    "10": 10,
    "A": 11,
}
_NONTRUMP_STRENGTH_BY_RANK = [_NONTRUMP_ORDER.index(name) for name in _RANK_NAMES]
_TRUMP_STRENGTH_BY_RANK = [_TRUMP_ORDER.index(name) for name in _RANK_NAMES]
_NONTRUMP_POINTS_BY_RANK = [_NONTRUMP_POINTS[name] for name in _RANK_NAMES]
_TRUMP_POINTS_BY_RANK = [_TRUMP_POINTS[name] for name in _RANK_NAMES]

# Cards are defined as 0, 1, ..., 31
PASS_ACTION = _NUM_CARDS  # 32
TAKE_ACTION = _NUM_CARDS + 1  # 33
CHOOSE_SUIT_ACTION_BASE = _NUM_CARDS + 2  # + suit index (0..3) : 34, 35, 36, 37

_GAME_TYPE = pyspiel.GameType(
    short_name="python_belote",
    long_name="Python Belote",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={
        "dealer": 0,
        "use_belote_rebelote": True,
        "max_redeals": _DEFAULT_MAX_REDEALS,
    },
)


def _make_game_info(max_redeals) -> pyspiel.GameInfo:
  """Creates GameInfo for the given `max_redeals` cap."""
  return pyspiel.GameInfo(
      # Card plays (0..31) + pass + take + 4 choose-suit actions.
      num_distinct_actions=_NUM_CARDS + 2 + _NUM_SUITS,
      max_chance_outcomes=_NUM_CARDS,
      num_players=_NUM_PLAYERS,
      # Loose bounds that also cover a capot (252 instead of 162) and the
      # optional belote/rebelote bonus.
      min_utility=-float(_MAX_SCORE_CAPOT + _BELOTE_REBELOTE_BONUS),
      max_utility=float(_MAX_SCORE_CAPOT + _BELOTE_REBELOTE_BONUS),
      utility_sum=0.0,
      # Each deal attempt is dealing (~32 draws) + bidding (up to 8 calls);
      # this can repeat up to max_redeals+1 times before the redeal cap
      # forces a flat draw, followed by card play (32 plays) if a deal
      # succeeds.
      max_game_length=(max_redeals + 1) * (_NUM_CARDS + 8) + _NUM_CARDS,
  )


_GAME_INFO = _make_game_info(_DEFAULT_MAX_REDEALS)


def card_suit(card) -> int:
    """Returns the suit index (0..3) of `card`."""
    return card // _NUM_RANKS


def card_rank_name(card) -> str:
    """Returns the rank name (e.g. "A") of `card`."""
    return _RANK_NAMES[card % _NUM_RANKS]


def card_string(card) -> str:
    """Returns the human-readable string (e.g. "AS") for `card`."""
    return card_rank_name(card) + _SUIT_NAMES[card_suit(card)]


def card_points(card, trump_suit) -> int:
    """Returns the point value of `card` given the current `trump_suit`."""
    rank = card % _NUM_RANKS
    return (
        _TRUMP_POINTS_BY_RANK[rank]
        if card_suit(card) == trump_suit
        else _NONTRUMP_POINTS_BY_RANK[rank]
    )


def card_strength(card, trump_suit) -> int:
    """Returns the relative ranking strength of `card` given `trump_suit`."""
    rank = card % _NUM_RANKS
    return (
        _TRUMP_STRENGTH_BY_RANK[rank]
        if card_suit(card) == trump_suit
        else _NONTRUMP_STRENGTH_BY_RANK[rank]
    )


def team_of(player) -> int:
    """Returns the team id (0 or 1) that `player` belongs to."""
    return player % 2


def partner_of(player) -> int:
    """Returns the id of `player`'s partner."""
    return (player + 2) % _NUM_PLAYERS


def _order_from(start) -> list[int]:
    return [(start + i) % _NUM_PLAYERS for i in range(_NUM_PLAYERS)]


def _initial_deal_schedule(dealer) -> list[int | None]:
    """Deal order for the first 5 cards/player (3 then 2) plus the turned card."""
    order = _order_from((dealer + 1) % _NUM_PLAYERS)
    schedule = []
    for player in order:
        schedule.extend([player] * 3)
    for player in order:
        schedule.extend([player] * 2)
    schedule.append(None)  # The next stock card is turned face up.
    return schedule


class BeloteGame(pyspiel.Game):
    """A Python version of Belote."""

    def __init__(self, params=None) -> None:
        params = params or {}
        max_redeals = params.get("max_redeals", _DEFAULT_MAX_REDEALS)
        super().__init__(_GAME_TYPE, _make_game_info(max_redeals), params)
        self.dealer = self.get_parameters().get("dealer", 0)
        self.use_belote_rebelote = self.get_parameters().get(
            "use_belote_rebelote", False
        )
        self.max_redeals = max_redeals

    def new_initial_state(self) -> "BeloteState":
        """Returns a state corresponding to the start of a game."""
        return BeloteState(self)

    def make_py_observer(self, iig_obs_type=None, params=None) -> "BeloteObserver":
        """Returns an object used for observing game state."""
        return BeloteObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
            params,
        )


class BeloteState(pyspiel.State):
    """A python version of the Belote state."""

    # Attribute count reflects the game's own state (deal/bid/play phases,
    # trick history, running scores, ...); splitting it up would not make
    # the logic clearer.
    # pylint: disable=too-many-instance-attributes

    def __init__(self, game) -> None:
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)
        self._dealer = game.dealer
        self.hands = [[] for _ in range(_NUM_PLAYERS)]
        self._deck = list(range(_NUM_CARDS))
        self._turned_card = None

        self._phase = "deal"
        self._deal_schedule = _initial_deal_schedule(self._dealer)
        self._deal_index = 0
        self._after_deal_phase = "bid1"

        self._bid_turn_order = _order_from((self._dealer + 1) % _NUM_PLAYERS)
        self._bid_pointer = 0

        self._max_redeals = game.max_redeals
        self._redeal_count = 0

        self._taker = -1
        self._trump_suit = -1
        self._declarer_team = -1
        self._use_belote_rebelote = game.use_belote_rebelote
        self._belote_player = -1

        self._trick = []
        self._trick_leader = -1
        self._current_player_play = -1
        self._tricks_played = 0
        self._played_cards = []
        self._trick_history = []
        self._trick_winners = []
        self._team_points = [0, 0]
        self._returns = [0.0] * _NUM_PLAYERS

    def current_player(self) -> int:
        """Returns id of the current player to act."""
        if self.is_terminal():
            return pyspiel.PlayerId.TERMINAL
        if self._phase == "deal":
            return pyspiel.PlayerId.CHANCE
        if self._phase in ("bid1", "bid2"):
            return self._bid_turn_order[self._bid_pointer]
        return self._current_player_play

    def _legal_actions(self, player) -> list[int]:
        """Returns a list of legal actions, sorted in ascending order."""
        assert player >= 0
        if self._phase == "bid1":
            return [PASS_ACTION, TAKE_ACTION]
        if self._phase == "bid2":
            turned_suit = card_suit(self._turned_card)
            return [PASS_ACTION] + [
                CHOOSE_SUIT_ACTION_BASE + s
                for s in range(_NUM_SUITS)
                if s != turned_suit
            ]
        if self._phase == "play":
            return self._legal_card_plays(player)
        return []

    # pylint: disable-next=too-many-return-statements
    def _legal_card_plays(self, player) -> list[int]:
        """Cards `player` may legally play, given suit/trump-following rules."""
        hand = self.hands[player]
        if not self._trick:  # No cards played for the trick, any card may be led.
            return sorted(hand)

        led_suit = card_suit(self._trick[0][1])
        trump = self._trump_suit
        same_suit_cards = [c for c in hand if card_suit(c) == led_suit]
        current_winner = self._trick_winner(self._trick)
        partner_winning = partner_of(player) == current_winner

        if same_suit_cards:
            if led_suit != trump:  # If the led suit is not trump, must follow suit.
                return sorted(same_suit_cards)
            # Trump was led: must play higher than the best trump so far if
            # possible, even if the partner currently holds the trick.
            highest = max(
                card_strength(c, trump) for _, c in self._trick if card_suit(c) == trump
            )
            higher = [c for c in same_suit_cards if card_strength(c, trump) > highest]
            return sorted(higher) if higher else sorted(same_suit_cards)

        # No cards of the led suit: may play trump if possible.
        trump_cards = [c for c in hand if card_suit(c) == trump]
        if trump_cards and led_suit != trump:
            if partner_winning:  # If the partner is currently winning, any card may be played.
                return sorted(hand)

            trumps_played = [c for _, c in self._trick if card_suit(c) == trump]
            if not trumps_played:  # No trumps have been played yet, play any trump.
                return sorted(trump_cards)

            # Need to play a higher trump if possible.
            highest = max(card_strength(c, trump) for c in trumps_played)
            higher = [c for c in trump_cards if card_strength(c, trump) > highest]
            return sorted(higher) if higher else sorted(trump_cards)

        # No cards of the led suit and no trumps: may play any card.
        return sorted(hand)

    def _is_better(self, card, other, led_suit, trump) -> bool:
        """Whether `card` beats `other` within the same trick."""
        card_trump = card_suit(card) == trump
        other_trump = card_suit(other) == trump

        # Exactly one card is trump, so `card` wins iff it is the trump card.
        if card_trump != other_trump:
            return card_trump

        # Both cards are trump, compare by trump ranking order
        if card_trump and other_trump:
            return card_strength(card, trump) > card_strength(other, trump)

        card_led = card_suit(card) == led_suit
        other_led = card_suit(other) == led_suit

        # Exactly one card follows the led suit, so `card` wins iff it follows it.
        if card_led != other_led:
            return card_led

        # Both cards follow led suit, compare by non-trump ranking order.
        if card_led and other_led:
            return card_strength(card, trump) > card_strength(other, trump)

        # Neither card is trump nor led suit: card cannot beat other.
        return False

    def _trick_winner(self, trick) -> int:
        """Returns the player currently winning `trick` (partial or complete)."""
        led_suit = card_suit(trick[0][1])
        best_player, best_card = trick[0]
        for player, card in trick[1:]:
            if self._is_better(card, best_card, led_suit, self._trump_suit):
                best_player, best_card = player, card

        return best_player

    def chance_outcomes(self) -> list[tuple[int, float]]:
        """Returns the possible chance outcomes and their probabilities."""
        assert self.is_chance_node()
        probability = 1.0 / len(self._deck)
        return [(card, probability) for card in self._deck]

    def _enter_play_phase(self) -> None:
        self._trick_leader = (self._dealer + 1) % _NUM_PLAYERS
        self._current_player_play = self._trick_leader
        self._trick = []
        self._tricks_played = 0
        self._team_points = [0, 0]
        if self._use_belote_rebelote:
            self._belote_player = self._find_belote_holder()

    def _trump_king_and_queen(self) -> tuple[int, int]:
        """Returns the (K, Q) card ids of the current trump suit."""
        return (
            self._trump_suit * _NUM_RANKS + _RANK_NAMES.index("K"),
            self._trump_suit * _NUM_RANKS + _RANK_NAMES.index("Q"),
        )

    def _find_belote_holder(self, hands=None, tricks=None) -> int:
        """Returns the id of the player who holds -- or, checked against
        `hands` other than `self.hands` (see `resample_from_infostate`), has
        already publicly played -- both K+Q of trump, or -1.

        Cards a player has already played are included alongside `hands`
        because who-played-what is public information regardless of which
        hand configuration `hands` represents: if a player is seen to play
        both the trump K and Q over the course of the game, that reveals the
        marriage even after the cards have left their hand, and that fact
        doesn't change across resampled worlds.
        """
        hands = self.hands if hands is None else hands
        tricks = self._reconstruct_tricks() if tricks is None else tricks
        trump_king, trump_queen = self._trump_king_and_queen()
        played_by = {p: set() for p in range(_NUM_PLAYERS)}
        for trick in tricks:
            for player, card in trick:
                played_by[player].add(card)
        for player, hand in enumerate(hands):
            combined = played_by[player] | set(hand)
            if trump_king in combined and trump_queen in combined:
                return player
        return -1


    def _apply_deal_action(self, card) -> None:
        self._deck.remove(card)
        destination = self._deal_schedule[self._deal_index]
        if destination is None:
            self._turned_card = card
        else:
            self.hands[destination].append(card)
        self._deal_index += 1
        if self._deal_index == len(self._deal_schedule):
            self._phase = self._after_deal_phase
            self._deal_schedule = []
            self._deal_index = 0
            if self._phase == "play":
                self._enter_play_phase()

    def _start_completion_deal(self, schedule, next_phase) -> None:
        self._deal_schedule = schedule
        self._deal_index = 0
        self._after_deal_phase = next_phase
        self._phase = "deal"

    def _completion_schedule_after_take(self, taker) -> list[int]:
        """3 cards to each non-taker, 2 to the taker (who already holds the turned card)."""
        order = _order_from((self._dealer + 1) % _NUM_PLAYERS)
        target_counts = {p: (2 if p == taker else 3) for p in order}
        dealt_counts = {p: 0 for p in order}
        schedule = []
        while any(dealt_counts[p] < target_counts[p] for p in order):
            for p in order:
                if dealt_counts[p] < target_counts[p]:
                    schedule.append(p)
                    dealt_counts[p] += 1
        return schedule

    def _apply_bid1_action(self, action, player) -> None:
        if action == TAKE_ACTION:
            self._taker = player
            self._trump_suit = card_suit(self._turned_card)
            self._declarer_team = team_of(player)
            self.hands[player].append(self._turned_card)
            self._start_completion_deal(
                self._completion_schedule_after_take(player), "play"
            )
        else:
            self._bid_pointer += 1
            if self._bid_pointer == _NUM_PLAYERS:
                self._phase = "bid2"
                self._bid_pointer = 0

    def _apply_bid2_action(self, action, player) -> None:
        if action == PASS_ACTION:
            self._bid_pointer += 1
            if self._bid_pointer == _NUM_PLAYERS:
                if self._redeal_count >= self._max_redeals:
                    # Redeal cap reached: rather than redealing forever, end
                    # the game here as a flat draw.
                    self._phase = "done"
                    self._returns = [0.0] * _NUM_PLAYERS
                    return
                # Everyone passed twice: reshuffle and redeal, dealer rotates.
                self._redeal_count += 1
                self._dealer = (self._dealer + 1) % _NUM_PLAYERS
                self.hands = [[] for _ in range(_NUM_PLAYERS)]
                self._turned_card = None
                self._deck = list(range(_NUM_CARDS))
                self._bid_turn_order = _order_from((self._dealer + 1) % _NUM_PLAYERS)
                self._bid_pointer = 0
                self._deal_schedule = _initial_deal_schedule(self._dealer)
                self._deal_index = 0
                self._after_deal_phase = "bid1"
                self._phase = "deal"
        else:
            suit = action - CHOOSE_SUIT_ACTION_BASE
            self._taker = player
            self._trump_suit = suit
            self._declarer_team = team_of(player)
            self.hands[player].append(self._turned_card)
            self._start_completion_deal(
                self._completion_schedule_after_take(player), "play"
            )

    def _finalize_scores(self):
        declarer_team = self._declarer_team
        other_team = 1 - declarer_team
        declarer_points = self._team_points[declarer_team]
        other_points = self._team_points[other_team]
        # 162 normally, or 252 if one team won all 8 tricks (capot).
        trick_total = declarer_points + other_points
        belote_team = team_of(self._belote_player) if self._belote_player >= 0 else -1
        declarer_bonus = _BELOTE_REBELOTE_BONUS if belote_team == declarer_team else 0
        other_bonus = _BELOTE_REBELOTE_BONUS if belote_team == other_team else 0
        # Contract success/failure is decided on totals that include the
        # belote/rebelote bonus, not on trick points alone.
        if declarer_points + declarer_bonus > other_points + other_bonus:
            final_declarer, final_other = declarer_points, other_points
        else:
            final_declarer, final_other = 0, trick_total
        final_declarer += declarer_bonus
        final_other += other_bonus
        diff = float(final_declarer - final_other)
        self._returns = [
            diff if team_of(p) == declarer_team else -diff for p in range(_NUM_PLAYERS)
        ]

    def _apply_play_action(self, card, player) -> None:
        self.hands[player].remove(card)
        self._trick.append((player, card))
        self._played_cards.append(card)
        if len(self._trick) < _NUM_PLAYERS:
            self._current_player_play = (player + 1) % _NUM_PLAYERS
            return

        winner = self._trick_winner(self._trick)
        points = sum(card_points(c, self._trump_suit) for _, c in self._trick)
        self._tricks_played += 1
        self._trick_winners.append(winner)
        if self._tricks_played == _NUM_CARDS // _NUM_PLAYERS:
            is_capot = all(team_of(w) == team_of(winner) for w in self._trick_winners)
            points += _CAPOT_LAST_TRICK_BONUS if is_capot else _LAST_TRICK_BONUS
        self._team_points[team_of(winner)] += points
        self._trick_history.append([c for _, c in self._trick])

        self._trick = []
        self._trick_leader = winner
        self._current_player_play = winner
        if self._tricks_played == _NUM_CARDS // _NUM_PLAYERS:
            self._finalize_scores()
            self._phase = "done"

    def _apply_action(self, action) -> None:
        """Applies an action and updates the state."""
        if self._phase == "deal":
            self._apply_deal_action(action)
        elif self._phase == "bid1":
            self._apply_bid1_action(action, self._bid_turn_order[self._bid_pointer])
        elif self._phase == "bid2":
            self._apply_bid2_action(action, self._bid_turn_order[self._bid_pointer])
        elif self._phase == "play":
            self._apply_play_action(action, self._current_player_play)

    def _action_to_string(self, player, action) -> str:
        """Action -> string."""
        if player == pyspiel.PlayerId.CHANCE:
            return f"Deal: {card_string(action)}"
        if action == PASS_ACTION:
            return "Pass"
        if action == TAKE_ACTION:
            return "Take"
        if CHOOSE_SUIT_ACTION_BASE <= action < CHOOSE_SUIT_ACTION_BASE + _NUM_SUITS:
            return f"Choose trump: {_SUIT_NAMES[action - CHOOSE_SUIT_ACTION_BASE]}"
        return f"Play: {card_string(action)}"

    def is_terminal(self) -> bool:
        """Returns True if the game is over."""
        return self._phase == "done"

    def returns(self) -> list[float]:
        """Total reward for each player over the course of the game so far."""
        return list(self._returns)

    def _reconstruct_tricks(self) -> list[list[tuple[int, int]]]:
        """Reconstructs the full trick history, including the current trick if any,
        as a list of lists of (player, card) pairs."""
        tricks = []
        leader = (self._dealer + 1) % _NUM_PLAYERS
        for i, cards in enumerate(self._trick_history):
            tricks.append(list(zip(_order_from(leader), cards)))
            leader = self._trick_winners[i]
        if self._trick:
            tricks.append(list(self._trick))
        return tricks

    def _infer_void_and_trump_bounds(
        self, tricks=None
    ) -> tuple[dict[int, set[int]], dict[int, int | None]]:
        """Infers, from the public information in the current state, which
        suits each player is known to be void in, and the maximum trump
        strength each player is known to hold (or None if no upper bound is
        known)."""
        trump = self._trump_suit
        void_suits = {p: set() for p in range(_NUM_PLAYERS)}
        max_trump_strength = {p: None for p in range(_NUM_PLAYERS)}
        if trump < 0: # No trump has been chosen yet.
            return void_suits, max_trump_strength

        for trick in (self._reconstruct_tricks() if tricks is None else tricks):
            if not trick:
                continue
            led_suit = card_suit(trick[0][1])
            for idx in range(1, len(trick)):
                player, card = trick[idx]
                suit = card_suit(card)
                partial = trick[:idx]
                current_winner = self._trick_winner(partial)
                partner_winning = partner_of(player) == current_winner
                trumps_played_before = [
                    c for _, c in partial if card_suit(c) == trump
                ]

                if suit != led_suit:
                    # Not following was only legal if void in that suit.
                    void_suits[player].add(trump if led_suit == trump else led_suit)
                    if suit != trump and led_suit != trump and not partner_winning:
                        # Trumping in was mandatory here too, so void there.
                        void_suits[player].add(trump)

                if suit == trump and trumps_played_before:
                    forced_to_overtrump = led_suit == trump or not partner_winning
                    highest = max(card_strength(c, trump) for c in trumps_played_before)
                    if forced_to_overtrump and card_strength(card, trump) <= highest:
                        # Didn't overtrump though forced to if possible: no
                        # trump above `highest` remains in hand.
                        bound = max_trump_strength[player]
                        if bound is None or highest < bound:
                            max_trump_strength[player] = highest

        return void_suits, max_trump_strength

    @staticmethod
    def _shuffle(items, sampler):
        """Shuffles `items` in place using the given `sampler` to drive
        randomness."""
        for i in range(len(items) - 1, 0, -1):
            j = int(sampler() * (i + 1))
            items[i], items[j] = items[j], items[i]

    @staticmethod
    def _bipartite_assign(unseen_cards, players, hand_sizes, allowed, sampler):
        """Partitions `unseen_cards` among `players` (matching `hand_sizes`)
        such that `allowed(p, c)` is True for every card `c` assigned to player `p`. 
        
        This is always possible because the constraints are derived
        from a real deal, and the algorithm is a randomized augmenting-path search 
        that finds a valid assignment in expected polynomial time.
        """
        assigned = {p: [] for p in players}
        player_order = list(players)
        BeloteState._shuffle(player_order, sampler)

        def try_place(card, visited_players):
            """Attempts to place `card` with one of the players, 
            possibly reassigning other cards to make room."""
            for p in player_order:
                if p in visited_players or not allowed(p, card):
                    continue
                visited_players.add(p)
                if len(assigned[p]) < hand_sizes[p]: # Player `p` has room for `card`.
                    assigned[p].append(card)
                    return True
                
                # No room: try to free up a slot by re-homing one of `p`'s
                # cards elsewhere (the augmenting-path step).
                bump_order = list(assigned[p])
                BeloteState._shuffle(bump_order, sampler)
                for other in bump_order: # Try to reassign `other` to another player.
                    assigned[p].remove(other)
                    if try_place(other, visited_players): # If `other` can be placed elsewhere, place `card` with `p`.
                        assigned[p].append(card)
                        return True
                    assigned[p].append(other)
            return False

        cards = list(unseen_cards)
        BeloteState._shuffle(cards, sampler)

        for card in cards:
            try_place(card, set())

        return assigned

    def _public_card_pins(self, player_id, hands) -> dict[int, list[int]]:
        """Returns a mapping from player id to the list of cards that are known 
        to be held by that player, based on public information (the turned card and any belote/rebelote cards). 
        
        The `hands` argument is used to check if a card is still in a player's hand."""
        pins = {}

        def pin(player, card):
            if player == player_id or card not in hands[player]:
                return
            cards = pins.setdefault(player, [])
            if card not in cards:  # turned card and belote card may coincide
                cards.append(card)

        # Pin the turned card to the taker, if any.
        if self._turned_card is not None and self._taker >= 0:
            pin(self._taker, self._turned_card)

        # Pin the other belote card to the belote holder, if any.
        if self._use_belote_rebelote and self._belote_player >= 0:
            trump_king, trump_queen = self._trump_king_and_queen()
            king_played = trump_king in self._played_cards
            queen_played = trump_queen in self._played_cards
            if king_played != queen_played:
                pin(self._belote_player, trump_queen if king_played else trump_king)

        return pins

    def _resample_unseen_cards(
        self, unseen_cards, other_players, hand_sizes, sampler, tricks
    ) -> dict[int, list[int]]:
        """Partitions `unseen_cards` among `other_players` (matching
        `hand_sizes`), respecting the void/trump-strength constraints from
        `_infer_void_and_trump_bounds`. See `_bipartite_assign` for why this
        is always satisfiable."""
        void_suits, max_trump_strength = self._infer_void_and_trump_bounds(tricks)
        trump = self._trump_suit

        def allowed(p, c):
            suit = card_suit(c)
            bound = max_trump_strength[p]
            return suit not in void_suits[p] and not (
                suit == trump and bound is not None and card_strength(c, trump) > bound
            )

        return self._bipartite_assign(
            unseen_cards, other_players, hand_sizes, allowed, sampler
        )

    def resample_from_infostate(self, player_id, sampler) -> "BeloteState":
        """Returns a clone with the other players' hands resampled, kept
        consistent with `player_id`'s information state: own hand and public
        history untouched, cards pinned by `_public_card_pins` kept with
        their known holder, and the rest resampled by
        `_resample_unseen_cards`. `sampler` is a zero-argument callable
        returning a uniform float in [0, 1), used to drive every shuffle so
        this respects the caller's RNG/seed.
        """
        clone = self.clone()
        other_players = [p for p in range(_NUM_PLAYERS) if p != player_id]

        tricks = self._reconstruct_tricks()

        pinned = self._public_card_pins(player_id, clone.hands)
        unseen_cards = []
        hand_sizes = {}
        for p in other_players:
            cards = [c for c in clone.hands[p] if c not in pinned.get(p, ())]
            unseen_cards.extend(cards)
            hand_sizes[p] = len(cards)

        assignment = self._resample_unseen_cards(
            unseen_cards, other_players, hand_sizes, sampler, tricks
        )
        for p, cards in pinned.items():
            assignment[p] = [*assignment[p], *cards]
        for p in other_players:
            clone.hands[p] = assignment[p]

        # If belote/rebelote is enabled, check if the resampled hands reveal a belote
        if clone._use_belote_rebelote:
            clone._belote_player = clone._find_belote_holder(clone.hands, tricks)

        return clone

    def __str__(self) -> str:
        """String for debug purposes. No particular semantics are required."""
        lines = [
            f"Dealer: {self._dealer}",
            f"Phase: {self._phase}",
            f"Hands: {[sorted(h) for h in self.hands]}",
        ]
        if self._turned_card is not None:
            lines.append(f"Turned card: {card_string(self._turned_card)}")
        if self._trump_suit >= 0:
            lines.append(
                f"Trump: {_SUIT_NAMES[self._trump_suit]}, Taker: {self._taker}"
            )
        if self._phase in ("play", "done"):
            lines.append(f"Trick: {[(p, card_string(c)) for p, c in self._trick]}")
            lines.append(f"Team points: {self._team_points}")
            if self._belote_player >= 0:
                lines.append(
                    f"Belote/rebelote holder: {self._belote_player}"
                    f" (team {team_of(self._belote_player)})"
                )
        return "\n".join(lines)


# BeloteObserver reads BeloteState's private fields directly, which is the
# established pattern for observers in this package (see e.g. ant_foraging.py).
# pylint: disable=protected-access
class BeloteObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, iig_obs_type, params=None) -> None:
        """Initializes an empty observation tensor."""
        del params
        self.iig_obs_type = iig_obs_type

        pieces = [("player", _NUM_PLAYERS, (_NUM_PLAYERS,))]
        if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
            pieces.append(("hand", _NUM_CARDS, (_NUM_CARDS,)))
        if iig_obs_type.public_info:
            pieces.append(("dealer", _NUM_PLAYERS, (_NUM_PLAYERS,)))
            pieces.append(("turned_card", _NUM_CARDS, (_NUM_CARDS,)))
            pieces.append(("trump_suit", _NUM_SUITS + 1, (_NUM_SUITS + 1,)))
            pieces.append(("declarer", _NUM_PLAYERS, (_NUM_PLAYERS,)))
            pieces.append(("current_trick", _NUM_CARDS, (_NUM_CARDS,)))
            pieces.append(("cards_played", _NUM_CARDS, (_NUM_CARDS,)))
            pieces.append(("team_points", 2, (2,)))
            if iig_obs_type.perfect_recall:
                num_tricks = _NUM_CARDS // _NUM_PLAYERS
                pieces.append(
                    ("trick_history", num_tricks * _NUM_CARDS, (num_tricks, _NUM_CARDS))
                )

        total_size = sum(size for name, size, shape in pieces)
        self.tensor = np.zeros(total_size, np.float32)
        self.dict = {}
        index = 0
        for name, size, shape in pieces:
            self.dict[name] = self.tensor[index : index + size].reshape(shape)
            index += size

    def set_from(self, state, player) -> None:  # pylint: disable=too-many-branches
        """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
        self.tensor.fill(0)
        if "player" in self.dict:
            self.dict["player"][player] = 1
        if "hand" in self.dict:
            for card in state.hands[player]:
                self.dict["hand"][card] = 1
        if "dealer" in self.dict:
            self.dict["dealer"][state._dealer] = 1
        if "turned_card" in self.dict and state._turned_card is not None:
            self.dict["turned_card"][state._turned_card] = 1
        if "trump_suit" in self.dict:
            index = state._trump_suit if state._trump_suit >= 0 else _NUM_SUITS
            self.dict["trump_suit"][index] = 1
        if "declarer" in self.dict and state._taker >= 0:
            self.dict["declarer"][state._taker] = 1
        if "current_trick" in self.dict:
            for _, card in state._trick:
                self.dict["current_trick"][card] = 1
        if "cards_played" in self.dict:
            for card in state._played_cards:
                self.dict["cards_played"][card] = 1
        if "team_points" in self.dict:
            self.dict["team_points"][0] = state._team_points[0] / float(
                _MAX_SCORE_CAPOT
            )
            self.dict["team_points"][1] = state._team_points[1] / float(
                _MAX_SCORE_CAPOT
            )
        if "trick_history" in self.dict:
            for trick_idx, cards in enumerate(state._trick_history):
                for card in cards:
                    self.dict["trick_history"][trick_idx][card] = 1

    def string_from(self, state, player) -> str:
        """Observation of `state` from the PoV of `player`, as a string."""
        pieces = []
        if "player" in self.dict:
            pieces.append(f"p{player}")
        if "hand" in self.dict:
            pieces.append(
                f"hand:{[card_string(c) for c in sorted(state.hands[player])]}"
            )
        if "dealer" in self.dict:
            pieces.append(f"dealer:{state._dealer}")
        if "turned_card" in self.dict and state._turned_card is not None:
            pieces.append(f"turned:{card_string(state._turned_card)}")
        if "trump_suit" in self.dict and state._trump_suit >= 0:
            pieces.append(f"trump:{_SUIT_NAMES[state._trump_suit]}")
        if "declarer" in self.dict and state._taker >= 0:
            pieces.append(f"declarer:{state._taker}")
        if "current_trick" in self.dict:
            pieces.append(f"trick:{[card_string(c) for _, c in state._trick]}")
        if "cards_played" in self.dict:
            pieces.append(f"played:{[card_string(c) for c in state._played_cards]}")
        if "trick_history" in self.dict and state._trick_history:
            pieces.append(
                "history:"
                + "|".join(
                    ",".join(card_string(c) for c in trick)
                    for trick in state._trick_history
                )
            )
        if "team_points" in self.dict:
            pieces.append(f"points:{state._team_points}")
        return " ".join(str(p) for p in pieces)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, BeloteGame)

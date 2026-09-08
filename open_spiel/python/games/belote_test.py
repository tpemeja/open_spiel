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
"""Tests for Python Belote."""

# Tests reach into BeloteState's private fields to set up and verify
# specific deals/tricks deterministically.
# pylint: disable=protected-access

import pickle

from absl.testing import absltest
import numpy as np

from open_spiel.python.games import belote
import pyspiel


def _deal_hands(state):
  """Applies chance actions until the initial 5-card deal + turned card."""
  while state.current_player(
  ) == pyspiel.PlayerId.CHANCE and state._phase == "deal":
    outcomes_with_probs = state.chance_outcomes()
    action_list, prob_list = zip(*outcomes_with_probs)
    action = np.random.choice(action_list, p=prob_list)
    state.apply_action(int(action))


def _finish_dealing(state):
  """Applies chance actions until a bidding phase is reached again."""
  while state.current_player() == pyspiel.PlayerId.CHANCE:
    outcomes_with_probs = state.chance_outcomes()
    action_list, prob_list = zip(*outcomes_with_probs)
    action = np.random.choice(action_list, p=prob_list)
    state.apply_action(int(action))


# Deals walked by test_consistent. Each is ~40 steps, and every step
# compares two full observation tensors per player, so this is the
# knob to turn if the test gets slow.
_CONSISTENCY_DEALS = 20


class BeloteTest(absltest.TestCase):
  """Tests for the BeloteGame and BeloteState classes."""

  def test_can_create_game_and_state(self):
    """Checks we can create the game and a state."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    self.assertEqual(state.current_player(), pyspiel.PlayerId.CHANCE)
    self.assertEqual(state.hands, [[] for _ in range(4)])

  def test_deal_and_turn_card(self):
    """Checks the initial deal gives 5 cards to each player and turns one up."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    _deal_hands(state)

    for hand in state.hands:
      self.assertLen(hand, 5)
    self.assertIsNotNone(state._turned_card)
    self.assertEqual(state._phase, "bid1")
    # Bidding starts to the left of the dealer.
    self.assertEqual(state.current_player(), (state._dealer + 1) % 4)
    self.assertCountEqual(state.legal_actions(),
                          [belote.PASS_ACTION, belote.TAKE_ACTION])

  def test_round1_take_completes_deal_to_eight_cards(self):
    """If a player takes in round 1, everyone ends up with 8 cards."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    _deal_hands(state)

    taker = state.current_player()
    turned_suit = belote.card_suit(state._turned_card)
    state.apply_action(belote.TAKE_ACTION)
    _finish_dealing(state)

    for hand in state.hands:
      self.assertLen(hand, 8)
    self.assertEqual(state._taker, taker)
    self.assertEqual(state._trump_suit, turned_suit)
    self.assertEqual(state._phase, "play")
    # Total cards in play must still be the full deck.
    all_cards = sorted(c for hand in state.hands for c in hand)
    self.assertEqual(all_cards, list(range(32)))

  def test_round2_choice_completes_deal_to_eight_cards(self):
    """If everyone passes round 1, round 2 lets players pick another suit."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    _deal_hands(state)
    turned_suit = belote.card_suit(state._turned_card)

    for _ in range(4):
      state.apply_action(belote.PASS_ACTION)

    self.assertEqual(state._phase, "bid2")
    legal = state.legal_actions()
    self.assertIn(belote.PASS_ACTION, legal)
    self.assertNotIn(belote.CHOOSE_SUIT_ACTION_BASE + turned_suit, legal)
    self.assertLen(legal, 4)  # pass + 3 remaining suits

    chosen_suit = next(s for s in range(4) if s != turned_suit)
    chooser = state.current_player()
    turned_card = state._turned_card
    state.apply_action(belote.CHOOSE_SUIT_ACTION_BASE + chosen_suit)
    _finish_dealing(state)

    for hand in state.hands:
      self.assertLen(hand, 8)
    self.assertEqual(state._taker, chooser)
    self.assertEqual(state._trump_suit, chosen_suit)
    self.assertEqual(state._phase, "play")
    # The player who picks the suit in round 2 keeps the turned card.
    self.assertIn(turned_card, state.hands[chooser])
    all_cards = sorted(c for hand in state.hands for c in hand)
    self.assertEqual(all_cards, list(range(32)))

  def test_all_pass_twice_redeals_and_rotates_dealer(self):
    """If everyone passes both rounds, the deal restarts with next dealer."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    original_dealer = state._dealer
    _deal_hands(state)
    for _ in range(8):  # 4 passes in round 1, 4 in round 2.
      state.apply_action(belote.PASS_ACTION)

    self.assertEqual(state._phase, "deal")
    self.assertEqual(state._dealer, (original_dealer + 1) % 4)
    self.assertEqual(state.hands, [[] for _ in range(4)])

  def test_redeal_cap_ends_game_as_flat_draw(self):
    """Once max_redeals is exceeded, the game ends instead of redealing."""
    game = belote.BeloteGame({"max_redeals": 1})
    state = game.new_initial_state()
    for _ in range(2):  # One full deal, then one redeal, both fully passed.
      _deal_hands(state)
      for _ in range(8):  # 4 passes in round 1, 4 in round 2.
        state.apply_action(belote.PASS_ACTION)

    self.assertTrue(state.is_terminal())
    self.assertEqual(state.returns(), [0.0] * 4)

  def test_bid_rounds_are_distinguishable_information_states(self):
    """A player's round-1 and round-2 decisions must be different
    information states. Both show the same 5 cards, the same turned card
    and no trump yet, so before the auction was added to the observer they
    were byte-identical -- a perfect-recall violation (the player could not
    remember having passed) that also made the two decisions, which offer
    different action sets, impossible to tell apart from the tensor."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    _deal_hands(state)
    player = state.current_player()

    bid1_tensor = list(state.information_state_tensor(player))
    bid1_string = state.information_state_string(player)
    bid1_actions = state.legal_actions(player)

    for _ in range(4):
      state.apply_action(belote.PASS_ACTION)

    self.assertEqual(state._phase, "bid2")
    self.assertEqual(state.current_player(), player)
    self.assertNotEqual(bid1_tensor, list(state.information_state_tensor(player)))
    self.assertNotEqual(bid1_string, state.information_state_string(player))
    self.assertNotEqual(bid1_actions, state.legal_actions(player))

  def test_auction_passes_are_recorded_in_bid_order(self):
    """Each pass is recorded against the player who made it, per round."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    _deal_hands(state)

    first, second = state._bid_turn_order[0], state._bid_turn_order[1]
    self.assertEqual(state._bid1_passes, [])
    state.apply_action(belote.PASS_ACTION)
    self.assertEqual(state._bid1_passes, [first])
    state.apply_action(belote.PASS_ACTION)
    self.assertEqual(state._bid1_passes, [first, second])

    for _ in range(2):  # Complete round 1.
      state.apply_action(belote.PASS_ACTION)
    self.assertEqual(state._bid2_passes, [])
    state.apply_action(belote.PASS_ACTION)
    self.assertEqual(state._bid1_passes, list(state._bid_turn_order))
    self.assertEqual(state._bid2_passes, [first])

  def test_auction_record_is_visible_to_every_player(self):
    """Passes are public: each player's own tensor reflects them, and the
    record survives into the play phase (who declined which suit stays
    informative evidence about the hands still held)."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    _deal_hands(state)
    observer = game.make_py_observer(
        pyspiel.IIGObservationType(perfect_recall=True))

    passer = state.current_player()
    state.apply_action(belote.PASS_ACTION)
    for player in range(4):
      observer.set_from(state, player)
      np.testing.assert_array_equal(
          observer.dict["bid1_passes"],
          [1.0 if p == passer else 0.0 for p in range(4)])

    taker = state.current_player()
    state.apply_action(belote.TAKE_ACTION)
    _finish_dealing(state)
    self.assertEqual(state._phase, "play")
    self.assertEqual(state._taker, taker)
    observer.set_from(state, taker)
    np.testing.assert_array_equal(
        observer.dict["bid1_passes"],
        [1.0 if p == passer else 0.0 for p in range(4)])
    np.testing.assert_array_equal(observer.dict["bid_round"], [1.0, 0.0, 0.0])

  def test_redeal_clears_the_auction_record(self):
    """A redeal deals brand-new hands, so the previous auction's passes say
    nothing about them and must not carry over."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    _deal_hands(state)
    for _ in range(8):  # 4 passes in round 1, 4 in round 2 -> redeal.
      state.apply_action(belote.PASS_ACTION)

    self.assertEqual(state._bid1_passes, [])
    self.assertEqual(state._bid2_passes, [])

  def test_must_follow_suit(self):
    """A player holding the led suit must play a card of that suit."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._phase = "play"
    state._trump_suit = 3  # Spades trump; leading suit below is Clubs.
    state._trick = [(0, 0)]  # Player 0 led the 7 of Clubs.
    state.hands[1] = [1, 8, 16]  # 8 of Clubs, 7 of Diamonds, 7 of Hearts.
    legal = state._legal_card_plays(1)
    self.assertEqual(legal, [1])  # Must follow with the Clubs card.

  def test_must_trump_when_void_and_opponent_winning(self):
    """A player void in the led suit must cut with trump if opponent leads."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._trump_suit = 3  # Spades.
    state._trick = [(0, 0)]  # Player 0 led 7 of Clubs; player 2 is partner.
    state.hands[1] = [16, 24]  # 7 of Hearts, 7 of Spades (trump).
    legal = state._legal_card_plays(1)
    self.assertEqual(legal, [24])  # Must cut with the only trump held.

  def test_no_obligation_to_trump_when_partner_winning(self):
    """No obligation to cut or overtrump while the partner holds the trick."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._trump_suit = 3  # Spades.
    state._trick = [(3, 0)]  # Partner (player 3) led and is winning.
    state.hands[1] = [16, 24]  # 7 of Hearts, 7 of Spades (trump).
    legal = state._legal_card_plays(1)
    self.assertCountEqual(legal, [16, 24])

  def test_must_overtrump_when_trump_led_even_if_partner_winning(self):
    """When trump itself is led, must overtrump even if partner is winning."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._trump_suit = 3  # Spades.
    # Player 3 (partner of player 1) led the 10 of Spades and currently wins.
    state._trick = [(3, 3 * 8 + 3)]
    state.hands[1] = [3 * 8 + 1, 3 * 8 + 4]  # 8 of Spades, Jack of Spades.
    legal = state._legal_card_plays(1)
    # Only the Jack of Spades outranks the 10 of Spades in the trump order;
    # the 8 of Spades does not, so it is forbidden despite the partner leading.
    self.assertEqual(legal, [3 * 8 + 4])

  def test_trick_winner_trump_beats_lead_suit(self):
    """A trump card always beats a non-trump card, regardless of lead suit."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._trump_suit = 3  # Spades.
    trick = [(0, 7), (1, 24), (2, 6),
             (3, 16)]  # Ace of Clubs vs 7 of Spades, etc.
    self.assertEqual(state._trick_winner(trick), 1)  # Trump always wins.

  def test_full_random_game_scores_correctly(self):
    """Plays a full random game and sanity-checks final scoring invariants."""
    game = belote.BeloteGame()
    for _ in range(20):
      state = game.new_initial_state()
      while not state.is_terminal():
        if state.is_chance_node():
          outcomes_with_probs = state.chance_outcomes()
          action_list, prob_list = zip(*outcomes_with_probs)
          action = np.random.choice(action_list, p=prob_list)
        else:
          legal = state.legal_actions()
          action = np.random.choice(legal)
        state.apply_action(int(action))

      returns = state.returns()
      self.assertAlmostEqual(sum(returns), 0.0)
      self.assertEqual(returns[0], returns[2])
      self.assertEqual(returns[1], returns[3])
      # 162 normally, or 252 in the (rare, random-play) case of a capot.
      self.assertIn(sum(state._team_points), (162, 252))

  def test_belote_rebelote_bonus_awarded_to_declarer(self):
    """The holder's team gets a 20-point bonus for holding K+Q of trump."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._trump_suit = 0  # Clubs.
    state.hands[0] = [6, 5]  # King and Queen of Clubs held by player 0.
    state._enter_play_phase()
    self.assertEqual(belote.team_of(state._belote_player), belote.team_of(0))

    state._declarer_team = 0
    state._team_points = [100, 62]
    state._finalize_scores()
    self.assertEqual(state.returns()[0], (100 + 20) - 62)

  def test_belote_rebelote_bonus_awarded_to_defenders(self):
    """The bonus goes to whichever team holds K+Q, even if defending."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._trump_suit = 0  # Clubs.
    state.hands[1] = [6, 5]  # King and Queen of Clubs held by player 1.
    state._enter_play_phase()
    self.assertEqual(belote.team_of(state._belote_player), belote.team_of(1))

    state._declarer_team = 0
    state._team_points = [100, 62]
    state._finalize_scores()
    self.assertEqual(state.returns()[0], 100 - (62 + 20))

  def test_belote_rebelote_bonus_survives_failed_contract(self):
    """The bonus is paid even if the holder's team scores 0 trick points."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._trump_suit = 0  # Clubs.
    state.hands[0] = [6, 5]  # King and Queen of Clubs held by player 0.
    state._enter_play_phase()
    self.assertEqual(belote.team_of(state._belote_player), belote.team_of(0))

    state._declarer_team = 0
    # Declarer team fails its contract outright: 0 trick points, so the
    # 162 trick points all go to the defenders, but the 20-point bonus
    # for holding K+Q of trump is unaffected by the failed contract.
    state._team_points = [0, 162]
    state._finalize_scores()
    self.assertEqual(state.returns()[0], 20 - 162)
    self.assertEqual(state.returns()[1], 162 - 20)

  def test_belote_rebelote_bonus_can_flip_a_failed_contract_to_success(self):
    """Official rule: the belote total counts toward contract success.

    75 raw trick points would fail the plain >81 threshold, but 75+20=95
    exceeds the defenders' 87, so the contract must succeed and the
    declarer keeps its 75 points (plus the 20-point bonus).
    """
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._trump_suit = 0  # Clubs.
    state.hands[0] = [6, 5]  # King and Queen of Clubs held by player 0.
    state._enter_play_phase()
    self.assertEqual(belote.team_of(state._belote_player), belote.team_of(0))

    state._declarer_team = 0
    state._team_points = [75, 87]
    state._finalize_scores()
    self.assertEqual(state.returns()[0], (75 + 20) - 87)

  def test_belote_rebelote_bonus_can_flip_a_made_contract_to_failure(self):
    """Symmetric case: the defenders' belote can sink an otherwise-made contract.

    The declarer's raw 85 trick points clear the plain >81 threshold, but
    the defenders hold belote (77+20=97 > 85), so the contract fails and
    the declarer keeps nothing.
    """
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._trump_suit = 0  # Clubs.
    state.hands[1] = [6, 5]  # King and Queen of Clubs held by player 1.
    state._enter_play_phase()
    self.assertEqual(belote.team_of(state._belote_player), belote.team_of(1))

    state._declarer_team = 0
    state._team_points = [85, 77]
    state._finalize_scores()
    self.assertEqual(state.returns()[0], 0 - (162 + 20))

  def test_belote_rebelote_requires_same_player_to_hold_both_cards(self):
    """Splitting K and Q of trump across partners does not earn the bonus."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._trump_suit = 0  # Clubs.
    state.hands[0] = [6]  # King of Clubs held by player 0.
    state.hands[2] = [5]  # Queen of Clubs held by partner (player 2).
    state._enter_play_phase()
    self.assertEqual(state._belote_player, -1)

  def test_full_random_game_with_belote_rebelote_scores_correctly(self):
    """Same sanity checks as above, exercising the belote/rebelote bonus."""
    game = belote.BeloteGame()
    for _ in range(20):
      state = game.new_initial_state()
      while not state.is_terminal():
        if state.is_chance_node():
          outcomes_with_probs = state.chance_outcomes()
          action_list, prob_list = zip(*outcomes_with_probs)
          action = np.random.choice(action_list, p=prob_list)
        else:
          legal = state.legal_actions()
          action = np.random.choice(legal)
        state.apply_action(int(action))

      returns = state.returns()
      self.assertAlmostEqual(sum(returns), 0.0)
      self.assertEqual(returns[0], returns[2])
      self.assertEqual(returns[1], returns[3])
      trick_total = sum(state._team_points)
      self.assertIn(trick_total, (162, 252))
      bonus = belote._BELOTE_REBELOTE_BONUS if state._belote_player >= 0 else 0
      self.assertLessEqual(abs(returns[0]), trick_total + bonus)

  def test_capot_awards_100_point_last_trick_bonus(self):
    """If one team wins all 8 tricks, the last trick is worth 100, not 10."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._trump_suit = 0  # Clubs; the final trick below is all Hearts.
    state._declarer_team = 0
    state._tricks_played = 7
    state._trick_winners = [0, 2, 0, 2, 0, 2, 0]  # Team 0 won every trick.
    state._team_points = [140, 12]

    # Final trick: player 0's Ace of Hearts (23) beats 8/9/10 of Hearts.
    state._trick = []
    state._trick_leader = 0
    state._current_player_play = 0
    state.hands[0] = [23]
    state.hands[1] = [17]
    state.hands[2] = [18]
    state.hands[3] = [19]
    state._apply_play_action(23, 0)
    state._apply_play_action(17, 1)
    state._apply_play_action(18, 2)
    state._apply_play_action(19, 3)

    # Card points in the trick: A=11, 8=0, 9=0, 10=10, i.e. 21, plus the
    # 100-point capot bonus (instead of the usual 10).
    self.assertEqual(state._team_points[0], 140 + 21 + 100)
    self.assertEqual(sum(state._team_points), 140 + 12 + 21 + 100)

  def test_non_capot_last_trick_keeps_10_point_bonus(self):
    """If tricks were split between teams, the last trick is worth only 10."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._trump_suit = 0  # Clubs; the final trick below is all Hearts.
    state._declarer_team = 0
    state._tricks_played = 7
    state._trick_winners = [0, 2, 0, 2, 0, 2, 1]  # Team 1 won one trick.
    state._team_points = [130, 22]

    state._trick = []
    state._trick_leader = 0
    state._current_player_play = 0
    state.hands[0] = [23]
    state.hands[1] = [17]
    state.hands[2] = [18]
    state.hands[3] = [19]
    state._apply_play_action(23, 0)
    state._apply_play_action(17, 1)
    state._apply_play_action(18, 2)
    state._apply_play_action(19, 3)

    self.assertEqual(state._team_points[0], 130 + 21 + 10)
    self.assertEqual(sum(state._team_points), 130 + 22 + 21 + 10)

  def test_capot_gives_defenders_252_points_on_failed_contract(self):
    """A capot changes the trick total used when the contract fails."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._declarer_team = 0
    # Team 1 (the defenders) achieved a capot: all 252 points are theirs.
    state._team_points = [0, 252]
    state._finalize_scores()
    self.assertEqual(state.returns()[0], 0 - 252)
    self.assertEqual(state.returns()[1], 252 - 0)

  def test_game_from_cc(self):
    """Runs the standard game tests, checking API consistency."""
    game = pyspiel.load_game("python_belote")
    pyspiel.random_sim_test(game, num_sims=10, serialize=False, verbose=False)

  def test_pickle(self):
    """Checks pickling and unpickling of game and state."""
    game = pyspiel.load_game("python_belote")
    pickled_game = pickle.dumps(game)
    unpickled_game = pickle.loads(pickled_game)
    self.assertEqual(str(game), str(unpickled_game))

    state = game.new_initial_state()
    _deal_hands(state)
    ser_str = pyspiel.serialize_game_and_state(game, state)
    new_game, new_state = pyspiel.deserialize_game_and_state(ser_str)
    self.assertEqual(str(game), str(new_game))
    self.assertEqual(str(state), str(new_state))
    pickled_state = pickle.dumps(state)
    unpickled_state = pickle.loads(pickled_state)
    self.assertEqual(str(state), str(unpickled_state))

  def test_cloned_state_matches_original_state(self):
    """Check we can clone states successfully."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    _deal_hands(state)
    clone = state.clone()

    self.assertEqual(state.history(), clone.history())
    self.assertEqual(state.num_players(), clone.num_players())
    self.assertEqual(state.move_number(), clone.move_number())
    self.assertEqual(state.num_distinct_actions(), clone.num_distinct_actions())
    self.assertEqual(state.hands, clone.hands)
    self.assertEqual(state._turned_card, clone._turned_card)

  def test_resample_from_infostate_keeps_own_hand_and_deck_consistent(self):
    """Resampling must never touch the requesting player's own hand, must
    preserve every hand's size, and the resampled hands plus already-played
    cards must still add up to exactly one full deck."""
    game = belote.BeloteGame()
    sampler = np.random.default_rng(0).random
    for _ in range(5):
      state = game.new_initial_state()
      while not state.is_terminal():
        if state.is_chance_node():
          outcomes_with_probs = state.chance_outcomes()
          action_list, prob_list = zip(*outcomes_with_probs)
          action = np.random.choice(action_list, p=prob_list)
        else:
          legal = state.legal_actions()
          action = np.random.choice(legal)
        state.apply_action(int(action))

        if state._phase == "play" and not state.is_terminal():
          for player_id in range(4):
            clone = state.resample_from_infostate(player_id, sampler)
            self.assertEqual(clone.hands[player_id], state.hands[player_id])
            self.assertEqual([len(h) for h in clone.hands],
                              [len(h) for h in state.hands])
            dealt = sorted(
                [c for hand in clone.hands for c in hand] +
                clone._played_cards)
            self.assertEqual(dealt, list(range(32)))

  def test_resample_from_infostate_respects_revealed_suit_void(self):
    """A player who couldn't follow the led suit (or trump) must never be
    dealt a card of that suit in a resampled hand."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._phase = "play"
    state._trump_suit = 3  # Spades.
    # Player 0 leads the 7 of Clubs; player 1 (void in Clubs, and forced to
    # trump since their partner isn't winning) discards the 7 of Hearts,
    # revealing they hold neither Clubs nor Spades (trump).
    state._trick = [(0, 0), (1, 16)]
    state.hands[0] = [1, 25, 9]  # 8C, 8S, 8D.
    state.hands[1] = [17, 10]  # 8H, 9D.
    state.hands[2] = [18, 19]
    state.hands[3] = [26, 2]  # 9S, 9C.

    void_suits, _ = state._infer_void_and_trump_bounds()
    self.assertEqual(void_suits[1], {0, 3})

    sampler = np.random.default_rng(1).random
    for _ in range(200):
      clone = state.resample_from_infostate(2, sampler)
      for card in clone.hands[1]:
        self.assertNotIn(belote.card_suit(card), (0, 3))

  def test_resample_from_infostate_respects_trump_strength_bound(self):
    """A player who didn't overtrump despite being forced to if possible
    must never be dealt a stronger trump than the one they let stand."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._phase = "play"
    state._trump_suit = 0  # Clubs.
    # Trump is led with the King (strength 3); player 1 follows with the 7
    # (strength 0) instead of overtrumping, revealing they hold no trump
    # stronger than the King.
    state._trick = [(0, 6), (1, 0)]
    state.hands[0] = [3, 7]  # 10C (strength 4), AC (strength 5).
    state.hands[1] = [8, 9]
    state.hands[2] = [20, 21]
    state.hands[3] = [16, 17]

    _, max_trump_strength = state._infer_void_and_trump_bounds()
    self.assertEqual(max_trump_strength[1], 3)

    sampler = np.random.default_rng(2).random
    for _ in range(200):
      clone = state.resample_from_infostate(2, sampler)
      for card in clone.hands[1]:
        if belote.card_suit(card) == 0:
          self.assertLessEqual(belote.card_strength(card, 0), 3)

  def test_resample_from_infostate_pins_turned_card_to_taker(self):
    """The turned card was picked up in full view of the table, so it must
    stay in the taker's hand in every resampled world."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._phase = "play"
    state._trump_suit = 0
    state._taker = 1
    state._turned_card = 5
    state.hands[1] = [5, 10, 11]
    state.hands[0] = [1, 2]
    state.hands[2] = [3, 4]
    state.hands[3] = [6, 7]

    sampler = np.random.default_rng(3).random
    for _ in range(200):
      clone = state.resample_from_infostate(0, sampler)
      self.assertIn(5, clone.hands[1])

  def test_resample_from_infostate_pins_belote_card_once_announced(self):
    """Official Belote requires announcing "belote" the moment the first of
    K+Q of trump is played, revealing the second is still held; the
    resample must keep it with that player in every resampled world."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._trump_suit = 0  # Clubs.
    state._dealer = 0
    state.hands[1] = [6, 5, 10]  # King and Queen of Clubs, plus one more.
    state.hands[0] = [1, 2]
    state.hands[2] = [3, 4]
    state.hands[3] = [7, 8]
    state._enter_play_phase()
    self.assertEqual(state._belote_player, 1)

    state._apply_play_action(6, 1)  # Player 1 plays the King of trump.
    self.assertNotIn(6, state.hands[1])
    self.assertIn(5, state.hands[1])  # Queen still held.

    sampler = np.random.default_rng(4).random
    for _ in range(200):
      clone = state.resample_from_infostate(0, sampler)
      self.assertIn(5, clone.hands[1])

  def test_resample_from_infostate_preserves_belote_holder_once_both_played(
      self):
    """Once both K and Q of trump have been publicly played by the same
    player, that's ground truth regardless of which world gets resampled."""
    game = belote.BeloteGame()
    state = game.new_initial_state()
    state._trump_suit = 0  # Clubs.
    state._dealer = 0
    # Player 1 led trick 0 with the King of trump, so players 0/2/3 (who
    # didn't follow with trump) are all publicly void in Clubs -- their
    # remaining hands below must stay consistent with that.
    state.hands[1] = [10]  # 9D.
    state.hands[0] = [16, 17]  # 7H, 8H.
    state.hands[2] = [18, 20]  # 9H, JH.
    state.hands[3] = [24, 25]  # 7S, 8S.
    # Trick 0: player 1 (King of Clubs) won it, so leads trick 1 too, where
    # they immediately play the Queen of Clubs.
    state._trick_history = [[6, 11, 19, 27]]
    state._trick_winners = [1]
    state._tricks_played = 1
    state._played_cards = [6, 11, 19, 27, 5]
    state._trick = [(1, 5)]
    state._belote_player = 1

    sampler = np.random.default_rng(5).random
    for player_id in (0, 2, 3):
      for _ in range(100):
        clone = state.resample_from_infostate(player_id, sampler)
        self.assertEqual(clone._belote_player, 1)

  def test_public_state_view_matches_internals(self):
    """The public accessors are the supported way to read a state. They must
    agree with the internals they front, or callers get a second, subtly
    different view of the same game."""
    game = pyspiel.load_game("python_belote")
    state = game.new_initial_state()
    rng = np.random.default_rng(11)
    while not state.is_terminal():
      self.assertEqual(state.current_phase(), state._phase)
      self.assertEqual(state.dealer(), state._dealer)
      self.assertEqual(state.upcard(), state._turned_card)
      self.assertEqual(state.taker(), state._taker)
      self.assertEqual(state.declarer_team(), state._declarer_team)
      self.assertEqual(state.trump_suit(), state._trump_suit)
      self.assertEqual(state.current_trick(), state._trick)
      self.assertEqual(state.trick_winners(), state._trick_winners)
      self.assertEqual(state.played_cards(), state._played_cards)
      self.assertEqual(state.team_points(), state._team_points)
      self.assertEqual(state.belote_holder(), state._belote_player)
      self.assertEqual(state.bid_passes(1), state._bid1_passes)
      self.assertEqual(state.bid_passes(2), state._bid2_passes)
      self.assertEqual(state.tricks(), state._reconstruct_tricks())
      if state._trump_suit < 0:
        self.assertIsNone(state.trump_marriage())
      else:
        self.assertEqual(state.trump_marriage(), state._trump_king_and_queen())
      if state.is_chance_node():
        outcomes, probs = zip(*state.chance_outcomes())
        state.apply_action(int(rng.choice(outcomes, p=probs)))
      else:
        state.apply_action(int(rng.choice(state.legal_actions())))

  def test_public_state_view_is_methods_not_properties(self):
    """OpenSpiel exposes a game-specific state accessor as a snake_case
    method -- `EuchreState::Upcard` reaches Python as `state.upcard()`,
    because it is a pybind-bound C++ getter. This game has a C++ twin, so a
    property here would mean a bot cannot be written against both."""
    state = pyspiel.load_game("python_belote").new_initial_state()
    for name in (
        "current_phase", "dealer", "upcard", "taker", "declarer_team",
        "trump_suit", "bidding_round", "bid_passes", "current_trick",
        "trick_winners", "played_cards", "team_points", "belote_holder",
        "belote_announced", "tricks", "trump_marriage", "public_inference",
    ):
      self.assertTrue(
          callable(getattr(state, name)), f"{name} must be a method")

  def test_trump_marriage_is_none_before_a_trump_exists(self):
    """As a private helper this could assume a trump suit; every caller ran
    after the auction. Public callers cannot be assumed to have checked, and
    the card ids computed from a trump of -1 are negative nonsense."""
    state = pyspiel.load_game("python_belote").new_initial_state()
    while state.is_chance_node():
      state.apply_action(state.legal_actions()[0])
    self.assertEqual(state.trump_suit(), -1)
    self.assertIsNone(state.trump_marriage())

    state.apply_action(belote.TAKE_ACTION)
    self.assertGreaterEqual(state.trump_suit(), 0)
    king, queen = state.trump_marriage()
    self.assertEqual(belote.card_suit(king), state.trump_suit())
    self.assertEqual(belote.card_suit(queen), state.trump_suit())
    self.assertEqual(belote.card_rank_name(king), "K")
    self.assertEqual(belote.card_rank_name(queen), "Q")

  def test_bidding_round_is_none_before_the_auction_opens(self):
    state = pyspiel.load_game("python_belote").new_initial_state()
    self.assertEqual(state.current_phase(), "deal")
    self.assertIsNone(state.bidding_round())

  def test_beats_is_a_module_level_function(self):
    """`beats` depends only on its arguments, so it belongs beside
    `card_points` and `card_strength` -- a strategy must be able to weigh a
    hypothetical trick without a state to hang the call on."""
    clubs, hearts = 0, 2
    seven_of_clubs = clubs * belote._NUM_RANKS + belote._RANK_NAMES.index("7")
    ace_of_hearts = hearts * belote._NUM_RANKS + belote._RANK_NAMES.index("A")
    # Clubs is trump: the 7 of trump beats the ace of a plain suit.
    self.assertTrue(
        belote.beats(seven_of_clubs, ace_of_hearts, hearts, clubs))
    self.assertFalse(
        belote.beats(ace_of_hearts, seven_of_clubs, hearts, clubs))

  def test_bid_passes_rejects_a_round_that_does_not_exist(self):
    state = pyspiel.load_game("python_belote").new_initial_state()
    with self.assertRaises(ValueError):
      state.bid_passes(3)

  def test_bidding_round_survives_the_auction_resolving(self):
    """`phase` stops describing the auction once it resolves, and reads
    "bid2" the moment round 2 opens even before anyone has acted.
    `bidding_round` has to stay correct through both."""
    game = pyspiel.load_game("python_belote")
    state = game.new_initial_state()
    while state.is_chance_node():
      state.apply_action(state.legal_actions()[0])
    self.assertEqual(state.current_phase(), "bid1")
    self.assertEqual(state.bidding_round(), 1)

    state.apply_action(belote.TAKE_ACTION)
    # The auction is over and phase has moved on, but it was won in round 1.
    self.assertNotIn(state.current_phase(), ("bid1", "bid2"))
    self.assertEqual(state.bidding_round(), 1)
    self.assertEqual(len(state.bid_passes(1)), 0)

  def test_bidding_round_reports_two_after_four_passes(self):
    game = pyspiel.load_game("python_belote")
    state = game.new_initial_state()
    while state.is_chance_node():
      state.apply_action(state.legal_actions()[0])
    for _ in range(belote._NUM_PLAYERS):
      state.apply_action(belote.PASS_ACTION)
    self.assertEqual(state.bidding_round(), 2)
    self.assertEqual(len(state.bid_passes(1)), belote._NUM_PLAYERS)

  def test_belote_announced_lags_belote_holder(self):
    """`belote_holder` is known to the engine from the deal; the holding only
    becomes public when a marriage card is played. A UI or an opponent model
    reading the former would be seeing information no player has."""
    game = pyspiel.load_game("python_belote")
    state = game.new_initial_state()
    rng = np.random.default_rng(3)
    while state.is_chance_node():
      outcomes, probs = zip(*state.chance_outcomes())
      state.apply_action(int(rng.choice(outcomes, p=probs)))
    state.apply_action(belote.TAKE_ACTION)
    while state.is_chance_node():
      outcomes, probs = zip(*state.chance_outcomes())
      state.apply_action(int(rng.choice(outcomes, p=probs)))

    if state.belote_holder() < 0:
      self.skipTest("this deal has no belote holding")
    # Holder known, nothing announced yet.
    self.assertEqual(state.belote_announced(), 0)

    king, queen = state.trump_marriage()
    seen = 0
    while not state.is_terminal():
      if state.is_chance_node():
        outcomes, probs = zip(*state.chance_outcomes())
        state.apply_action(int(rng.choice(outcomes, p=probs)))
        continue
      action = int(rng.choice(state.legal_actions()))
      state.apply_action(action)
      if action in (king, queen):
        seen += 1
      self.assertEqual(state.belote_announced(), seen)
    self.assertEqual(seen, 2)

  def test_beats_agrees_with_trick_winners(self):
    """`beats` is the rules comparison strategies use to reason about a
    trick; it must agree with how the engine actually awards them."""
    game = pyspiel.load_game("python_belote")
    rng = np.random.default_rng(7)
    for _ in range(20):
      state = game.new_initial_state()
      while not state.is_terminal():
        if state.is_chance_node():
          outcomes, probs = zip(*state.chance_outcomes())
          state.apply_action(int(rng.choice(outcomes, p=probs)))
        else:
          state.apply_action(int(rng.choice(state.legal_actions())))
      trump = state.trump_suit()
      if trump < 0:
        continue
      for trick, winner in zip(state.tricks(), state.trick_winners()):
        led_suit = belote.card_suit(trick[0][1])
        best_player, best_card = trick[0]
        for player, card in trick[1:]:
          if belote.beats(card, best_card, led_suit, trump):
            best_player, best_card = player, card
        self.assertEqual(best_player, winner)

  def test_public_inference_matches_resampling_constraints(self):
    """`public_inference` is what resampling uses to keep a world legal.
    Anything featurising or sampling should be able to rely on the same
    answer the engine relies on."""
    game = pyspiel.load_game("python_belote")
    rng = np.random.default_rng(13)
    state = game.new_initial_state()
    while not state.is_terminal():
      if state.is_chance_node():
        outcomes, probs = zip(*state.chance_outcomes())
        state.apply_action(int(rng.choice(outcomes, p=probs)))
        continue
      if state.current_phase() == "play" and len(state.trick_winners()) >= 2:
        voids, _ = state.public_inference()
        player = state.current_player()
        world = state.resample_from_infostate(player, rng.random)
        for seat, suits in voids.items():
          if seat == player:
            continue
          for card in world.hands[seat]:
            self.assertNotIn(belote.card_suit(card), suits)
      state.apply_action(int(rng.choice(state.legal_actions())))


  def test_consistent(self):
    """Checks the Python and C++ game implementations are the same.

    `kuhn_poker_test` does this by enumerating every state, which belote's
    state space rules out, so this walks random deals instead and compares
    both games at every step: what each offers, what each shows a player, and
    the public view each exposes.
    """
    py_game = pyspiel.load_game("python_belote")
    cc_game = pyspiel.load_game("belote")
    self.assertEqual(py_game.information_state_tensor_size(),
                     cc_game.information_state_tensor_size())
    self.assertEqual(py_game.observation_tensor_size(),
                     cc_game.observation_tensor_size())

    for deal in range(_CONSISTENCY_DEALS):
      rng = np.random.default_rng(deal)
      py_state = py_game.new_initial_state()
      cc_state = cc_game.new_initial_state()
      while not py_state.is_terminal():
        where = f"deal {deal}, history {py_state.history()}"
        self.assertEqual(py_state.is_chance_node(), cc_state.is_chance_node(),
                         where)
        self.assertEqual(py_state.current_player(), cc_state.current_player(),
                         where)
        self.assertEqual(py_state.legal_actions(), cc_state.legal_actions(),
                         where)
        self._assert_public_view_agrees(py_state, cc_state, where)
        for player in range(belote._NUM_PLAYERS):
          self.assertEqual(py_state.information_state_string(player),
                           cc_state.information_state_string(player), where)
          self.assertEqual(py_state.observation_string(player),
                           cc_state.observation_string(player), where)
          np.testing.assert_array_equal(
              py_state.information_state_tensor(player),
              cc_state.information_state_tensor(player), where)
          np.testing.assert_array_equal(
              py_state.observation_tensor(player),
              cc_state.observation_tensor(player), where)

        if py_state.is_chance_node():
          py_outcomes = py_state.chance_outcomes()
          self.assertEqual(py_outcomes, cc_state.chance_outcomes(), where)
          outcomes, probs = zip(*py_outcomes)
          action = int(rng.choice(outcomes, p=probs))
        else:
          action = int(rng.choice(py_state.legal_actions()))
        py_state.apply_action(action)
        cc_state.apply_action(action)

      self.assertTrue(cc_state.is_terminal(), f"deal {deal}")
      self.assertEqual(py_state.returns(), cc_state.returns(), f"deal {deal}")

  def _assert_public_view_agrees(self, py_state, cc_state, where):
    """Every accessor added for agents must read the same on both games."""
    self.assertEqual(py_state.current_phase(), cc_state.current_phase(), where)
    self.assertEqual(py_state.dealer(), cc_state.dealer(), where)
    self.assertEqual(py_state.upcard(), cc_state.upcard(), where)
    self.assertEqual(py_state.taker(), cc_state.taker(), where)
    self.assertEqual(py_state.declarer_team(), cc_state.declarer_team(), where)
    self.assertEqual(py_state.trump_suit(), cc_state.trump_suit(), where)
    self.assertEqual(py_state.bidding_round(), cc_state.bidding_round(), where)
    self.assertEqual(py_state.bid_passes(1), cc_state.bid_passes(1), where)
    self.assertEqual(py_state.bid_passes(2), cc_state.bid_passes(2), where)
    self.assertEqual([tuple(pair) for pair in py_state.current_trick()],
                     [tuple(pair) for pair in cc_state.current_trick()], where)
    self.assertEqual(list(py_state.trick_winners()),
                     list(cc_state.trick_winners()), where)
    self.assertEqual(list(py_state.played_cards()),
                     list(cc_state.played_cards()), where)
    self.assertEqual(list(py_state.team_points()),
                     list(cc_state.team_points()), where)
    self.assertEqual(py_state.belote_holder(), cc_state.belote_holder(), where)
    self.assertEqual(py_state.belote_announced(), cc_state.belote_announced(),
                     where)
    self.assertEqual(py_state.trump_marriage(), cc_state.trump_marriage(),
                     where)
    self.assertEqual([sorted(hand) for hand in py_state.hands],
                     [sorted(hand) for hand in cc_state.hands], where)
    self.assertEqual(
        [[tuple(pair) for pair in trick] for trick in py_state.tricks()],
        [[tuple(pair) for pair in trick] for trick in cc_state.tricks()], where)
    self.assertEqual(py_state.public_inference(), cc_state.public_inference(),
                     where)

  def test_beats_matches_the_cc_implementation(self):
    """`beats` is duplicated in C++; a strategy must rank a trick the same
    way whichever game it is pointed at."""
    for trump in range(belote._NUM_SUITS):
      for led_suit in range(belote._NUM_SUITS):
        for card in range(belote._NUM_CARDS):
          for other in range(belote._NUM_CARDS):
            self.assertEqual(
                belote.beats(card, other, led_suit, trump),
                pyspiel.belote.beats(card, other, led_suit, trump),
                f"beats({card}, {other}, {led_suit}, {trump})")


if __name__ == "__main__":
  absltest.main()

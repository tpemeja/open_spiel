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
"""Tests for the game-specific functions for belote."""

import pickle

from absl.testing import absltest

import pyspiel

belote = pyspiel.belote


class GamesBeloteTest(absltest.TestCase):

  def test_bindings(self):
    self.assertEqual(belote.NUM_PLAYERS, 4)
    self.assertEqual(belote.NUM_SUITS, 4)
    self.assertEqual(belote.NUM_RANKS, 8)
    self.assertEqual(belote.NUM_CARDS, 32)
    self.assertEqual(belote.NUM_TRICKS, 8)
    self.assertEqual(belote.PASS_ACTION, 32)
    self.assertEqual(belote.TAKE_ACTION, 33)
    self.assertEqual(belote.CHOOSE_SUIT_ACTION_BASE, 34)
    self.assertEqual(belote.BELOTE_REBELOTE_BONUS, 20)
    self.assertEqual(belote.LAST_TRICK_BONUS, 10)
    self.assertEqual(belote.CAPOT_LAST_TRICK_BONUS, 100)

    game = pyspiel.load_game('belote')
    state = game.new_initial_state()
    self.assertEqual(state.current_phase(), belote.Phase.DEAL)
    self.assertEqual(state.dealer(), 0)
    self.assertIsNone(state.upcard())
    self.assertEqual(state.taker(), -1)
    self.assertEqual(state.declarer_team(), -1)
    self.assertEqual(state.trump_suit(), -1)
    self.assertIsNone(state.bidding_round())
    self.assertEqual(state.bid_passes(1), [])
    self.assertEqual(state.bid_passes(2), [])
    self.assertEqual(state.current_trick(), [])
    self.assertEqual(state.trick_winners(), [])
    self.assertEqual(state.played_cards(), [])
    self.assertEqual(state.team_points(), [0, 0])
    self.assertEqual(state.belote_holder(), -1)
    self.assertEqual(state.belote_announced(), 0)
    self.assertEqual(state.tricks(), [])
    self.assertIsNone(state.trump_marriage())
    self.assertEqual(state.hands(), [[], [], [], []])
    no_voids = {p: set() for p in range(4)}
    no_bounds = {p: None for p in range(4)}
    self.assertEqual(state.public_inference(), (no_voids, no_bounds))

    self.assertEqual(belote.card_suit(8), 1)
    self.assertEqual(belote.card_rank(8), 0)
    self.assertEqual(belote.card_string(8), '7D')
    self.assertEqual(belote.card_points(6, 0), 4)
    self.assertEqual(belote.team_of(2), 0)
    self.assertEqual(belote.partner_of(1), 3)
    # Clubs is trump, diamonds was led: the lowest trump beats the ace of a
    # plain suit. card_strength ranks within a suit and so cannot express
    # that on its own -- beats() is what knows trump outranks the led suit.
    seven_of_clubs, ace_of_diamonds = 0, 15
    self.assertTrue(belote.beats(seven_of_clubs, ace_of_diamonds, 1, 0))
    self.assertFalse(belote.beats(ace_of_diamonds, seven_of_clubs, 1, 0))
    # Within trump, belote ranks the Jack top, above the Ace.
    jack_of_clubs, ace_of_clubs = 4, 7
    self.assertGreater(
        belote.card_strength(jack_of_clubs, 0),
        belote.card_strength(ace_of_clubs, 0))
    self.assertTrue(belote.beats(jack_of_clubs, ace_of_clubs, 0, 0))

  def test_accessors_track_a_played_deal(self):
    """The accessors exist for agents mid-deal, not only at the root."""
    state = pyspiel.load_game('belote').new_initial_state()
    while state.is_chance_node():
      state.apply_action(state.legal_actions()[0])
    self.assertEqual(state.current_phase(), belote.Phase.BID1)
    self.assertEqual(state.bidding_round(), 1)
    self.assertIsNotNone(state.upcard())

    state.apply_action(belote.TAKE_ACTION)
    self.assertGreaterEqual(state.taker(), 0)
    self.assertEqual(state.declarer_team(), belote.team_of(state.taker()))
    self.assertGreaterEqual(state.trump_suit(), 0)
    # Taken in round 1, and phase has already moved past the auction.
    self.assertEqual(state.bidding_round(), 1)
    self.assertEqual(state.bid_passes(1), [])

    king, queen = state.trump_marriage()
    self.assertEqual(belote.card_suit(king), state.trump_suit())
    self.assertEqual(belote.card_suit(queen), state.trump_suit())

    while state.is_chance_node():
      state.apply_action(state.legal_actions()[0])
    self.assertEqual(state.current_phase(), belote.Phase.PLAY)
    self.assertEqual([len(hand) for hand in state.hands()], [8, 8, 8, 8])

    state.apply_action(state.legal_actions()[0])
    self.assertLen(state.current_trick(), 1)
    self.assertLen(state.played_cards(), 1)
    self.assertLen(state.tricks(), 1)

  def test_pickle_round_trip(self):
    """The state and game pickle bindings must survive a round trip."""
    state = pyspiel.load_game('belote').new_initial_state()
    while state.is_chance_node():
      state.apply_action(state.legal_actions()[0])
    state.apply_action(belote.TAKE_ACTION)
    while state.is_chance_node():
      state.apply_action(state.legal_actions()[0])
    state.apply_action(state.legal_actions()[0])

    unpickled = pickle.loads(pickle.dumps(state))
    self.assertEqual(str(unpickled), str(state))
    self.assertEqual(unpickled.history(), state.history())
    self.assertEqual(unpickled.hands(), state.hands())
    self.assertEqual(unpickled.trump_suit(), state.trump_suit())
    self.assertEqual(unpickled.taker(), state.taker())

  def test_cloned_state_matches_original(self):
    state = pyspiel.load_game('belote').new_initial_state()
    while state.is_chance_node():
      state.apply_action(state.legal_actions()[0])
    clone = state.clone()
    self.assertEqual(str(clone), str(state))
    self.assertEqual(clone.history(), state.history())
    self.assertEqual(clone.hands(), state.hands())

  def test_score_deal_binding(self):
    """The scoring rule is bound as a free function, like beats()."""
    # Made contract: each side keeps what it took.
    self.assertEqual(
        belote.score_deal(91, 71, belote.BeloteSide.NONE), (91, 71))
    # A tie fails the contract, so the defenders collect all 162.
    self.assertEqual(
        belote.score_deal(81, 81, belote.BeloteSide.NONE), (0, 162))
    # The 20-point bonus counts toward the threshold and can flip a deal.
    self.assertEqual(
        belote.score_deal(75, 87, belote.BeloteSide.DECLARERS), (95, 87))
    self.assertEqual(
        belote.score_deal(85, 77, belote.BeloteSide.DEFENDERS), (0, 182))

  def test_invalid_parameters_are_rejected(self):
    for params in ('dealer=-1', 'dealer=4'):
      with self.assertRaises(pyspiel.SpielError, msg=params):
        pyspiel.load_game(f'belote({params})')

  def test_bid_passes_rejects_a_round_that_does_not_exist(self):
    state = pyspiel.load_game('belote').new_initial_state()
    with self.assertRaises(pyspiel.SpielError):
      state.bid_passes(3)


if __name__ == '__main__':
  absltest.main()

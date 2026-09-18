// Copyright 2019 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/python/pybind11/games_belote.h"

#include <memory>
#include <string>
#include <utility>

#include "open_spiel/games/belote/belote.h"
#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/spiel.h"

// Upcard(), BiddingRound() and TrumpMarriage() return absl::optional, so the
// absl casters are needed to map absl::nullopt onto Python's None.
#include "pybind11/include/pybind11/detail/common.h"
#include "pybind11_abseil/absl_casters.h"

namespace open_spiel {

namespace py = ::pybind11;
using belote::BeloteGame;
using belote::BeloteState;

void init_pyspiel_games_belote(py::module& m) {
  py::module_ belote = m.def_submodule("belote");

  belote.attr("NUM_PLAYERS") = py::int_(belote::kNumPlayers);
  belote.attr("NUM_SUITS") = py::int_(belote::kNumSuits);
  belote.attr("NUM_RANKS") = py::int_(belote::kNumRanks);
  belote.attr("NUM_CARDS") = py::int_(belote::kNumCards);
  belote.attr("NUM_TRICKS") = py::int_(belote::kNumTricks);
  belote.attr("PASS_ACTION") = py::int_(belote::kPassAction);
  belote.attr("TAKE_ACTION") = py::int_(belote::kTakeAction);
  belote.attr("CHOOSE_SUIT_ACTION_BASE") =
      py::int_(belote::kChooseSuitActionBase);
  belote.attr("BELOTE_REBELOTE_BONUS") =
      py::int_(belote::kBeloteRebeloteBonus);
  belote.attr("LAST_TRICK_BONUS") = py::int_(belote::kLastTrickBonus);
  belote.attr("CAPOT_LAST_TRICK_BONUS") =
      py::int_(belote::kCapotLastTrickBonus);

  py::enum_<belote::Phase>(belote, "Phase")
      .value("DEAL", belote::Phase::kDeal)
      .value("BID1", belote::Phase::kBid1)
      .value("BID2", belote::Phase::kBid2)
      .value("PLAY", belote::Phase::kPlay)
      .value("GAME_OVER", belote::Phase::kGameOver)
      .export_values();

  py::enum_<belote::BeloteSide>(belote, "BeloteSide")
      .value("NONE", belote::BeloteSide::kNone)
      .value("DECLARERS", belote::BeloteSide::kDeclarers)
      .value("DEFENDERS", belote::BeloteSide::kDefenders)
      .export_values();

  belote.def("card_string", belote::CardString);
  belote.def("card_suit", belote::CardSuit);
  belote.def("card_rank", belote::CardRank);
  belote.def("card_points", belote::CardPoints);
  belote.def("card_strength", belote::CardStrength);
  belote.def("team_of", belote::TeamOf);
  belote.def("partner_of", belote::PartnerOf);
  // Free functions, not state accessors: they depend only on their arguments,
  // so a strategy can weigh a hypothetical trick or contract with no state to
  // hang the call on.
  belote.def("beats", belote::Beats);
  belote.def("score_deal", belote::ScoreDeal);

  py::classh<BeloteState, State> state_class(belote, "BeloteState");
  state_class
      .def("current_phase", &BeloteState::CurrentPhase)
      .def("dealer", &BeloteState::Dealer)
      .def("upcard", &BeloteState::Upcard)
      .def("taker", &BeloteState::Taker)
      .def("declarer_team", &BeloteState::DeclarerTeam)
      .def("trump_suit", &BeloteState::TrumpSuit)
      .def("bidding_round", &BeloteState::BiddingRound)
      .def("bid_passes", &BeloteState::BidPasses, py::arg("round_number"))
      .def("current_trick", &BeloteState::CurrentTrick)
      .def("trick_winners", &BeloteState::TrickWinners)
      .def("played_cards", &BeloteState::PlayedCards)
      .def("team_points", &BeloteState::TeamPoints)
      .def("belote_holder", &BeloteState::BeloteHolder)
      .def("belote_announced", &BeloteState::BeloteAnnounced)
      .def("tricks", &BeloteState::Tricks)
      .def("trump_marriage", &BeloteState::TrumpMarriage)
      // Returns a copy, so mutating the result does not change the state.
      .def("hands", &BeloteState::PlayerHands)
      // Returns (void_suits, max_trump_strength): a dict of seat -> set of
      // suits, and a dict of seat -> bound or None (VoidAndTrumpBounds
      // spells "no bound" as -1).
      .def("public_inference",
           [](const BeloteState& state) {
             belote::VoidAndTrumpBounds bounds = state.PublicInference();
             py::dict void_suits;
             py::dict max_trump_strength;
             for (Player player = 0; player < belote::kNumPlayers; ++player) {
               py::set suits;
               for (int suit = 0; suit < belote::kNumSuits; ++suit) {
                 if (bounds.void_suits[player][suit]) {
                   suits.add(py::int_(suit));
                 }
               }
               void_suits[py::int_(player)] = suits;
               int bound = bounds.max_trump_strength[player];
               max_trump_strength[py::int_(player)] =
                   bound < 0 ? py::none() : py::object(py::int_(bound));
             }
             return py::make_tuple(void_suits, max_trump_strength);
           })
      // Pickle support
      .def(py::pickle(
          [](const BeloteState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<BeloteState*>(game_and_state.second.release());
          }));

  py::classh<BeloteGame, Game>(m, "BeloteGame")
      // Pickle support
      .def(py::pickle(
          [](std::shared_ptr<const BeloteGame> game) {  // __getstate__
            return game->ToString();
          },
          [](const std::string& data) {  // __setstate__
            return std::dynamic_pointer_cast<BeloteGame>(
                std::const_pointer_cast<Game>(LoadGame(data)));
          }));
}

}  // namespace open_spiel

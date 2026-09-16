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

#include "open_spiel/games/belote/belote.h"

#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace belote {
namespace {

void BasicGameTests() {
  testing::LoadGameTest("belote");
  testing::ChanceOutcomesTest(*LoadGame("belote"));
  testing::RandomSimTest(*LoadGame("belote"), 100);
}

// Random-simulate many games and check invariants that must hold regardless
// of the (random) trump choice and card play: partners always score the
// same, and the two teams' returns sum to zero.
void ManyRandomGamesInvariantsTest() {
  std::shared_ptr<const Game> game = LoadGame("belote");
  std::mt19937 rng(98765);
  for (int i = 0; i < 2000; ++i) {
    std::unique_ptr<State> state = game->NewInitialState();
    int num_actions = 0;
    while (!state->IsTerminal()) {
      std::vector<Action> legal_actions = state->LegalActions();
      SPIEL_CHECK_FALSE(legal_actions.empty());
      Action action;
      if (state->IsChanceNode()) {
        std::vector<std::pair<Action, double>> outcomes =
            state->ChanceOutcomes();
        action = SampleAction(outcomes, rng).first;
      } else {
        std::uniform_int_distribution<int> dis(0, legal_actions.size() - 1);
        action = legal_actions[dis(rng)];
      }
      state->ApplyAction(action);
      ++num_actions;
      SPIEL_CHECK_LE(num_actions, game->MaxGameLength());
    }
    std::vector<double> returns = state->Returns();
    SPIEL_CHECK_EQ(returns.size(), kNumPlayers);
    SPIEL_CHECK_EQ(returns[0], returns[2]);
    SPIEL_CHECK_EQ(returns[1], returns[3]);
    SPIEL_CHECK_FLOAT_EQ(returns[0] + returns[1], 0.0);
  }
}

// If all 8 calls (4 in each round) are passes, the deal is thrown in and the
// game ends as a draw -- there is no redeal.
void AllPassEndsTheGameAsADrawTest() {
  std::shared_ptr<const Game> game = LoadGame("belote");
  std::unique_ptr<State> state = game->NewInitialState();
  while (state->IsChanceNode()) state->ApplyAction(state->LegalActions()[0]);
  for (int i = 0; i < 2 * kNumPlayers; ++i) {
    SPIEL_CHECK_FALSE(state->IsTerminal());
    state->ApplyAction(kPassAction);
  }
  SPIEL_CHECK_TRUE(state->IsTerminal());
  SPIEL_CHECK_EQ(state->Returns(), std::vector<double>(kNumPlayers, 0.0));
}

// Applies chance actions (the first legal one each time, for determinism)
// until the initial 5-card-per-player + turned-card deal is complete and the
// state has reached the round-1 bidding phase.
void DealInitialHands(State* state) {
  while (state->IsChanceNode()) {
    state->ApplyAction(state->LegalActions()[0]);
  }
}

// A player's round-1 and round-2 decisions must be different information
// states. Both show the same 5 cards, the same turned card and no trump yet,
// so before bid_round was added to the tensor they were identical -- a
// perfect-recall violation (the player could not remember having passed)
// that also made the two decisions, which offer different action sets,
// impossible to tell apart from the tensor.
void BidRoundsAreDistinguishableInformationStatesTest() {
  std::shared_ptr<const Game> game = LoadGame("belote");
  std::unique_ptr<State> state = game->NewInitialState();
  DealInitialHands(state.get());
  Player player = state->CurrentPlayer();

  std::vector<float> bid1_tensor = state->InformationStateTensor(player);
  std::string bid1_string = state->InformationStateString(player);
  std::vector<Action> bid1_actions = state->LegalActions();

  for (int i = 0; i < 4; ++i) state->ApplyAction(kPassAction);

  SPIEL_CHECK_EQ(state->CurrentPlayer(), player);
  SPIEL_CHECK_TRUE(bid1_tensor != state->InformationStateTensor(player));
  SPIEL_CHECK_TRUE(bid1_string != state->InformationStateString(player));
  SPIEL_CHECK_TRUE(bid1_actions != state->LegalActions());
}

// Each pass is recorded against the player who made it, per round, and is
// public: every player's information state reflects it.
void AuctionPassesAreRecordedInBidOrderTest() {
  std::shared_ptr<const Game> game = LoadGame("belote");
  std::unique_ptr<State> state = game->NewInitialState();
  DealInitialHands(state.get());

  Player first = state->CurrentPlayer();
  state->ApplyAction(kPassAction);
  for (Player p = 0; p < kNumPlayers; ++p) {
    SPIEL_CHECK_TRUE(absl::StrContains(
        state->InformationStateString(p),
        absl::StrCat("passed1:[", first, "]")));
  }

  Player second = state->CurrentPlayer();
  state->ApplyAction(kPassAction);
  SPIEL_CHECK_TRUE(absl::StrContains(
      state->InformationStateString(first),
      absl::StrCat("passed1:[", first, ", ", second, "]")));

  for (int i = 0; i < 2; ++i) state->ApplyAction(kPassAction);  // Round 1.
  SPIEL_CHECK_FALSE(
      absl::StrContains(state->InformationStateString(first), "passed2:"));
  Player third = state->CurrentPlayer();
  state->ApplyAction(kPassAction);  // Round 2.
  SPIEL_CHECK_TRUE(absl::StrContains(
      state->InformationStateString(first),
      absl::StrCat("passed2:[", third, "]")));
}

// Resampling must never change what `p` can already see: their own hand and
// all public information (dealer, trump, tricks, points, ...), captured
// here via information-state equality, must be identical before and after.
void ResampleFromInfostateTest() {
  std::shared_ptr<const Game> game = LoadGame("belote");
  std::mt19937 rng(12345);
  UniformProbabilitySampler sampler;
  int num_sims = 100;
  for (int sim = 0; sim < num_sims; ++sim) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      if (!state->IsChanceNode()) {
        for (int p = 0; p < state->NumPlayers(); ++p) {
          std::unique_ptr<State> resampled =
              state->ResampleFromInfostate(p, sampler);
          SPIEL_CHECK_EQ(state->InformationStateTensor(p),
                         resampled->InformationStateTensor(p));
          SPIEL_CHECK_EQ(state->InformationStateString(p),
                         resampled->InformationStateString(p));
          SPIEL_CHECK_EQ(state->CurrentPlayer(), resampled->CurrentPlayer());
        }
      }
      if (state->IsChanceNode()) {
        std::vector<std::pair<Action, double>> outcomes =
            state->ChanceOutcomes();
        state->ApplyAction(SampleAction(outcomes, rng).first);
        continue;
      }
      std::vector<Action> actions = state->LegalActions();
      std::uniform_int_distribution<int> dis(0, actions.size() - 1);
      state->ApplyAction(actions[dis(rng)]);
    }
  }
}

// Plays random games, calling `visit` on every card-play decision.
void ForEachRandomPlayState(
    int num_games, int seed,
    const std::function<void(const BeloteState&)>& visit) {
  std::shared_ptr<const Game> game = LoadGame("belote");
  std::mt19937 rng(seed);
  for (int i = 0; i < num_games; ++i) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        state->ApplyAction(SampleAction(state->ChanceOutcomes(), rng).first);
        continue;
      }
      const auto& belote_state = static_cast<const BeloteState&>(*state);
      if (belote_state.CurrentPhase() == Phase::kPlay) visit(belote_state);
      std::vector<Action> actions = state->LegalActions();
      std::uniform_int_distribution<int> dis(0, actions.size() - 1);
      state->ApplyAction(actions[dis(rng)]);
    }
  }
}

// Belote is announced to every player the moment the holder plays the first
// of the trump King and Queen -- and not before, since until then the holding
// is private.
void BeloteAnnouncementIsPublicTest() {
  int num_announced = 0;
  int num_unannounced_holders = 0;
  ForEachRandomPlayState(300, 2468, [&](const BeloteState& state) {
    const int ann_offset = kNumPlayers + kNumCards + kNumPlayers + kNumCards +
                           (kNumSuits + 1) + kNumPlayers + 3 +
                           kNumPlayers * kNumCards + kNumCards + 2;
    bool announced = state.BeloteAnnounced() > 0;
    if (announced) ++num_announced;
    if (state.BeloteHolder() >= 0 && !announced) ++num_unannounced_holders;
    std::string expected = absl::StrCat(" belote:", state.BeloteHolder(), " ");
    for (Player p = 0; p < kNumPlayers; ++p) {
      SPIEL_CHECK_EQ(absl::StrContains(state.InformationStateString(p),
                                       " belote:"),
                     announced);
      SPIEL_CHECK_EQ(absl::StrContains(state.ObservationString(p), " belote:"),
                     announced);
      if (announced) {
        SPIEL_CHECK_TRUE(
            absl::StrContains(state.InformationStateString(p), expected));
      }
      std::vector<float> observation = state.State::ObservationTensor(p);
      std::vector<float> infostate = state.State::InformationStateTensor(p);
      for (Player seat = 0; seat < kNumPlayers; ++seat) {
        float bit = announced && seat == state.BeloteHolder() ? 1 : 0;
        SPIEL_CHECK_EQ(observation[ann_offset + seat], bit);
        SPIEL_CHECK_EQ(infostate[ann_offset + seat], bit);
      }
    }
  });
  SPIEL_CHECK_GT(num_announced, 0);
  SPIEL_CHECK_GT(num_unannounced_holders, 0);
}

// Once exactly one of the trump King and Queen has been played, whether it
// was announced is public, and resampling must follow it both ways: the
// other card stays with an announced holder, and never lands with a seat
// that played its partner card without announcing.
void ResampleFollowsBeloteAnnouncementTest() {
  UniformProbabilitySampler sampler;
  int num_pinned = 0;
  int num_barred = 0;
  ForEachRandomPlayState(1000, 97531, [&](const BeloteState& state) {
    const auto [king, queen] = *state.TrumpMarriage();
    std::vector<int> played = state.PlayedCards();
    bool king_played = absl::c_linear_search(played, king);
    bool queen_played = absl::c_linear_search(played, queen);
    if (king_played == queen_played) return;
    int played_card = king_played ? king : queen;
    int other_card = king_played ? queen : king;
    Player played_by = kInvalidPlayer;
    for (const auto& trick : state.Tricks()) {
      for (const auto& [p, c] : trick) {
        if (c == played_card) played_by = p;
      }
    }
    SPIEL_CHECK_GE(played_by, 0);
    bool announced = state.BeloteAnnounced() > 0;
    SPIEL_CHECK_EQ(announced, state.BeloteHolder() == played_by);
    for (Player observer = 0; observer < kNumPlayers; ++observer) {
      if (observer == played_by) continue;
      for (int i = 0; i < 5; ++i) {
        std::unique_ptr<State> resampled =
            state.ResampleFromInfostate(observer, sampler);
        std::vector<int> hand = static_cast<const BeloteState&>(*resampled)
                                    .PlayerHands()[played_by];
        SPIEL_CHECK_EQ(absl::c_linear_search(hand, other_card), announced);
        SPIEL_CHECK_EQ(state.InformationStateString(observer),
                       resampled->InformationStateString(observer));
      }
      ++(announced ? num_pinned : num_barred);
    }
  });
  SPIEL_CHECK_GT(num_pinned, 0);
  SPIEL_CHECK_GT(num_barred, 0);
}

}  // namespace
}  // namespace belote
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::belote::BasicGameTests();
  open_spiel::belote::ManyRandomGamesInvariantsTest();
  open_spiel::belote::AllPassEndsTheGameAsADrawTest();
  open_spiel::belote::BidRoundsAreDistinguishableInformationStatesTest();
  open_spiel::belote::AuctionPassesAreRecordedInBidOrderTest();
  open_spiel::belote::ResampleFromInfostateTest();
  open_spiel::belote::BeloteAnnouncementIsPublicTest();
  open_spiel::belote::ResampleFollowsBeloteAnnouncementTest();
}

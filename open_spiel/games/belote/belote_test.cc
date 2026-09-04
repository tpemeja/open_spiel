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

#include <memory>
#include <random>
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
  testing::RandomSimTest(*LoadGame("belote(max_redeals=0)"), 100);
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
      SPIEL_CHECK_LE(num_actions, 1500);
    }
    std::vector<double> returns = state->Returns();
    SPIEL_CHECK_EQ(returns.size(), kNumPlayers);
    SPIEL_CHECK_EQ(returns[0], returns[2]);
    SPIEL_CHECK_EQ(returns[1], returns[3]);
    SPIEL_CHECK_FLOAT_EQ(returns[0] + returns[1], 0.0);
  }
}

// A redeal cap of 0 forces the very first failed bidding round (all 8 passes
// across bid1+bid2) to end the game as a flat draw instead of redealing.
void MaxRedealsFlatDrawTest() {
  std::shared_ptr<const Game> game = LoadGame("belote(max_redeals=0)");
  std::mt19937 rng(13579);
  bool saw_flat_draw = false;
  for (int i = 0; i < 200 && !saw_flat_draw; ++i) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        std::vector<std::pair<Action, double>> outcomes =
            state->ChanceOutcomes();
        state->ApplyAction(SampleAction(outcomes, rng).first);
        continue;
      }
      std::vector<Action> legal_actions = state->LegalActions();
      // Always pass to force a flat draw as soon as possible.
      if (absl::c_linear_search(legal_actions, kPassAction)) {
        state->ApplyAction(kPassAction);
      } else {
        state->ApplyAction(legal_actions[0]);
      }
    }
    std::vector<double> returns = state->Returns();
    if (absl::c_all_of(returns, [](double r) { return r == 0.0; })) {
      saw_flat_draw = true;
    }
  }
  SPIEL_CHECK_TRUE(saw_flat_draw);
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

// A redeal deals brand-new hands, so the previous auction's passes say
// nothing about them and must not carry over.
void RedealClearsTheAuctionRecordTest() {
  std::shared_ptr<const Game> game = LoadGame("belote");
  std::unique_ptr<State> state = game->NewInitialState();
  DealInitialHands(state.get());

  for (int i = 0; i < 8; ++i) {  // 4 passes in round 1, 4 in round 2.
    state->ApplyAction(kPassAction);
  }

  SPIEL_CHECK_TRUE(state->IsChanceNode());  // Redealt, not a flat draw.
  std::string info = state->InformationStateString(0);
  SPIEL_CHECK_FALSE(absl::StrContains(info, "passed1:"));
  SPIEL_CHECK_FALSE(absl::StrContains(info, "passed2:"));
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

}  // namespace
}  // namespace belote
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::belote::BasicGameTests();
  open_spiel::belote::ManyRandomGamesInvariantsTest();
  open_spiel::belote::MaxRedealsFlatDrawTest();
  open_spiel::belote::BidRoundsAreDistinguishableInformationStatesTest();
  open_spiel::belote::AuctionPassesAreRecordedInBidOrderTest();
  open_spiel::belote::RedealClearsTheAuctionRecordTest();
  open_spiel::belote::ResampleFromInfostateTest();
}

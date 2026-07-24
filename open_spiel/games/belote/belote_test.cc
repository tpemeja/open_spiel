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
  testing::RandomSimTest(*LoadGame("belote(use_belote_rebelote=false)"), 100);
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

}  // namespace
}  // namespace belote
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::belote::BasicGameTests();
  open_spiel::belote::ManyRandomGamesInvariantsTest();
  open_spiel::belote::MaxRedealsFlatDrawTest();
}

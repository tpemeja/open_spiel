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
#include <utility>
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

// The card rankings and point values, pinned against the published rules
// (pagat.com): trump runs J 9 A 10 K Q 8 7 and plain suits run A 10 K Q J 9
// 8 7, and the 32 cards are worth 152 points plus 10 for the last trick.
void CardValuesMatchTheRulesTest() {
  // Rank indices, in the order kRankNames declares them.
  enum { k7, k8, k9, k10, kJ, kQ, kK, kA };
  const int trump = 0;  // Clubs; the tables are the same for any trump.
  auto trump_card = [&](int rank) { return trump * kNumRanks + rank; };
  auto plain_card = [&](int rank) { return 1 * kNumRanks + rank; };

  // Strongest first.
  const int trump_order[] = {kJ, k9, kA, k10, kK, kQ, k8, k7};
  const int trump_points[] = {20, 14, 11, 10, 4, 3, 0, 0};
  const int plain_order[] = {kA, k10, kK, kQ, kJ, k9, k8, k7};
  const int plain_points[] = {11, 10, 4, 3, 2, 0, 0, 0};

  for (int i = 0; i < kNumRanks; ++i) {
    SPIEL_CHECK_EQ(CardPoints(trump_card(trump_order[i]), trump),
                   trump_points[i]);
    SPIEL_CHECK_EQ(CardPoints(plain_card(plain_order[i]), trump),
                   plain_points[i]);
    if (i + 1 < kNumRanks) {
      // Each card outranks the next one in the list, within its own suit.
      SPIEL_CHECK_GT(CardStrength(trump_card(trump_order[i]), trump),
                     CardStrength(trump_card(trump_order[i + 1]), trump));
      SPIEL_CHECK_GT(CardStrength(plain_card(plain_order[i]), trump),
                     CardStrength(plain_card(plain_order[i + 1]), trump));
      SPIEL_CHECK_TRUE(Beats(trump_card(trump_order[i]),
                             trump_card(trump_order[i + 1]), trump, trump));
      SPIEL_CHECK_TRUE(Beats(plain_card(plain_order[i]),
                             plain_card(plain_order[i + 1]), 1, trump));
    }
  }

  // Any trump beats any plain card, whichever suit was led.
  for (int t = 0; t < kNumRanks; ++t) {
    for (int c = 0; c < kNumRanks; ++c) {
      SPIEL_CHECK_TRUE(Beats(trump_card(t), plain_card(c), 1, trump));
      SPIEL_CHECK_FALSE(Beats(plain_card(c), trump_card(t), 1, trump));
    }
  }

  // The whole deck is worth 152 in cards: 62 in trump, 30 in each plain suit.
  int deck_total = 0;
  for (int card = 0; card < kNumCards; ++card) {
    deck_total += CardPoints(card, trump);
  }
  SPIEL_CHECK_EQ(deck_total, kMaxScore - kLastTrickBonus);
  SPIEL_CHECK_EQ(deck_total + kLastTrickBonus, 162);
  SPIEL_CHECK_EQ(deck_total + kCapotLastTrickBonus, 252);
}

// ScoreDeal is a pure function, so each scoring rule can be pinned exactly
// instead of waiting for a random deal to produce the right point split.
void ScoreDealTest() {
  // Made contract: each side keeps what it took (91 + 71 = 162).
  SPIEL_CHECK_EQ(ScoreDeal(91, 71, BeloteSide::kNone),
                 std::make_pair(91, 71));
  // Failed contract: the defenders collect every trick point.
  SPIEL_CHECK_EQ(ScoreDeal(71, 91, BeloteSide::kNone),
                 std::make_pair(0, 162));
  // A tie is a failure: the declarers must score strictly more. That holds
  // once the bonus is added too, so 91 against 71 + 20 is a failed contract.
  SPIEL_CHECK_EQ(ScoreDeal(81, 81, BeloteSide::kNone),
                 std::make_pair(0, 162));
  SPIEL_CHECK_EQ(ScoreDeal(91, 71, BeloteSide::kDefenders),
                 std::make_pair(0, 182));

  // The belote bonus is credited to its holders on a made contract...
  SPIEL_CHECK_EQ(ScoreDeal(91, 71, BeloteSide::kDeclarers),
                 std::make_pair(111, 71));
  SPIEL_CHECK_EQ(ScoreDeal(100, 62, BeloteSide::kDefenders),
                 std::make_pair(100, 82));
  // ...and on a failed one, where the declarers keep the 20 and nothing else.
  SPIEL_CHECK_EQ(ScoreDeal(71, 91, BeloteSide::kDeclarers),
                 std::make_pair(20, 162));

  // The bonus counts toward the threshold, so it can flip a deal either way.
  // 75 < 87 on tricks alone, but 75 + 20 > 87 makes the contract.
  SPIEL_CHECK_EQ(ScoreDeal(75, 87, BeloteSide::kDeclarers),
                 std::make_pair(95, 87));
  // 85 > 77 on tricks alone, but 77 + 20 > 85 sinks it.
  SPIEL_CHECK_EQ(ScoreDeal(85, 77, BeloteSide::kDefenders),
                 std::make_pair(0, 182));

  // Capot: the deck is worth 252, and it all goes to whoever scores higher.
  SPIEL_CHECK_EQ(ScoreDeal(kMaxScoreCapot, 0, BeloteSide::kNone),
                 std::make_pair(kMaxScoreCapot, 0));
  SPIEL_CHECK_EQ(ScoreDeal(0, kMaxScoreCapot, BeloteSide::kNone),
                 std::make_pair(0, kMaxScoreCapot));
}

// The trick points of a finished deal always add up to 162, or 252 when one
// team took every trick, and Returns() is ScoreDeal applied to that split.
void TerminalScoringTest() {
  std::shared_ptr<const Game> game = LoadGame("belote");
  std::mt19937 rng(31415);
  int num_capots = 0;
  int num_normal = 0;
  int num_with_belote = 0;
  for (int i = 0; i < 3000; ++i) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (!state->IsTerminal()) {
      if (state->IsChanceNode()) {
        state->ApplyAction(SampleAction(state->ChanceOutcomes(), rng).first);
        continue;
      }
      std::vector<Action> actions = state->LegalActions();
      std::uniform_int_distribution<int> dis(0, actions.size() - 1);
      state->ApplyAction(actions[dis(rng)]);
    }
    const auto& belote_state = static_cast<const BeloteState&>(*state);
    // A deal everyone passed out of scores nothing and has no trump.
    if (belote_state.TrumpSuit() < 0) {
      SPIEL_CHECK_EQ(state->Returns(), std::vector<double>(kNumPlayers, 0.0));
      continue;
    }
    std::vector<int> points = belote_state.TeamPoints();
    std::vector<Player> winners = belote_state.TrickWinners();
    SPIEL_CHECK_EQ(winners.size(), kNumTricks);
    bool capot = absl::c_all_of(winners, [&](Player w) {
      return TeamOf(w) == TeamOf(winners[0]);
    });
    SPIEL_CHECK_EQ(points[0] + points[1], capot ? kMaxScoreCapot : kMaxScore);
    if (capot) {
      ++num_capots;
      // The capot side took everything; the other side took nothing.
      SPIEL_CHECK_EQ(points[1 - TeamOf(winners[0])], 0);
    } else {
      ++num_normal;
    }

    int declarer_team = belote_state.DeclarerTeam();
    Player belote_holder = belote_state.BeloteHolder();
    BeloteSide side = BeloteSide::kNone;
    if (belote_holder >= 0) {
      ++num_with_belote;
      side = TeamOf(belote_holder) == declarer_team ? BeloteSide::kDeclarers
                                                    : BeloteSide::kDefenders;
    }
    const auto [declarer, defender] = ScoreDeal(
        points[declarer_team], points[1 - declarer_team], side);
    double diff = static_cast<double>(declarer - defender);
    std::vector<double> returns = state->Returns();
    for (Player p = 0; p < kNumPlayers; ++p) {
      SPIEL_CHECK_FLOAT_EQ(returns[p],
                           TeamOf(p) == declarer_team ? diff : -diff);
    }
  }
  // All three situations must actually have come up, or the sweep above is
  // only checking the easy case.
  SPIEL_CHECK_GT(num_capots, 0);
  SPIEL_CHECK_GT(num_normal, 0);
  SPIEL_CHECK_GT(num_with_belote, 0);
}

// Taking in round 1 or naming a suit in round 2 both complete the deal to 8
// cards each, with the turned card in the taker's hand.
void TakingCompletesTheDealTest() {
  std::shared_ptr<const Game> game = LoadGame("belote");
  std::mt19937 rng(2718);
  for (bool take_in_round_one : {true, false}) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (state->IsChanceNode()) {
      state->ApplyAction(SampleAction(state->ChanceOutcomes(), rng).first);
    }
    const auto& belote_state = static_cast<const BeloteState&>(*state);
    SPIEL_CHECK_EQ(belote_state.CurrentPhase(), Phase::kBid1);
    int upcard = *belote_state.Upcard();
    Player taker;
    if (take_in_round_one) {
      taker = state->CurrentPlayer();
      state->ApplyAction(kTakeAction);
      SPIEL_CHECK_EQ(belote_state.TrumpSuit(), CardSuit(upcard));
    } else {
      for (int i = 0; i < kNumPlayers; ++i) state->ApplyAction(kPassAction);
      SPIEL_CHECK_EQ(belote_state.CurrentPhase(), Phase::kBid2);
      taker = state->CurrentPlayer();
      // Any suit other than the turned one; the first legal call will do.
      Action call = state->LegalActions().back();
      state->ApplyAction(call);
      SPIEL_CHECK_EQ(belote_state.TrumpSuit(), call - kChooseSuitActionBase);
      SPIEL_CHECK_NE(belote_state.TrumpSuit(), CardSuit(upcard));
    }
    SPIEL_CHECK_EQ(belote_state.Taker(), taker);
    SPIEL_CHECK_EQ(belote_state.DeclarerTeam(), TeamOf(taker));
    while (state->IsChanceNode()) {
      state->ApplyAction(SampleAction(state->ChanceOutcomes(), rng).first);
    }
    SPIEL_CHECK_EQ(belote_state.CurrentPhase(), Phase::kPlay);
    for (const std::vector<int>& hand : belote_state.PlayerHands()) {
      SPIEL_CHECK_EQ(hand.size(), kNumTricks);
    }
    // The turned card ends up with the taker.
    SPIEL_CHECK_TRUE(
        absl::c_linear_search(belote_state.PlayerHands()[taker], upcard));
  }
}

// BiddingRound() keeps reporting the round that resolved the contract, and
// reports 2 once round 1 has been passed out.
void BiddingRoundAccessorTest() {
  std::shared_ptr<const Game> game = LoadGame("belote");
  std::mt19937 rng(4242);
  std::unique_ptr<State> state = game->NewInitialState();
  const auto& belote_state = static_cast<const BeloteState&>(*state);
  SPIEL_CHECK_FALSE(belote_state.BiddingRound().has_value());
  while (state->IsChanceNode()) {
    state->ApplyAction(SampleAction(state->ChanceOutcomes(), rng).first);
  }
  SPIEL_CHECK_EQ(*belote_state.BiddingRound(), 1);
  for (int i = 0; i < kNumPlayers; ++i) state->ApplyAction(kPassAction);
  // Round 1 passed out: now in round 2, with all four passes on record.
  SPIEL_CHECK_EQ(*belote_state.BiddingRound(), 2);
  SPIEL_CHECK_EQ(belote_state.BidPasses(1).size(), kNumPlayers);
  SPIEL_CHECK_TRUE(belote_state.BidPasses(2).empty());
  state->ApplyAction(state->LegalActions().back());
  // Resolved in round 2, and it still says so after the auction is over.
  SPIEL_CHECK_EQ(*belote_state.BiddingRound(), 2);
  while (state->IsChanceNode()) {
    state->ApplyAction(SampleAction(state->ChanceOutcomes(), rng).first);
  }
  SPIEL_CHECK_EQ(belote_state.CurrentPhase(), Phase::kPlay);
  SPIEL_CHECK_EQ(*belote_state.BiddingRound(), 2);
}

// The belote/rebelote bonus needs one player holding both marriage cards;
// split between partners it counts for nobody.
void BeloteNeedsASingleHolderTest() {
  std::shared_ptr<const Game> game = LoadGame("belote");
  std::mt19937 rng(8888);
  int num_held = 0;
  int num_split = 0;
  for (int i = 0; i < 500; ++i) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (state->IsChanceNode()) {
      state->ApplyAction(SampleAction(state->ChanceOutcomes(), rng).first);
    }
    // Take in round 1 so there is always a trump and a full deal.
    state->ApplyAction(kTakeAction);
    while (state->IsChanceNode()) {
      state->ApplyAction(SampleAction(state->ChanceOutcomes(), rng).first);
    }
    const auto& belote_state = static_cast<const BeloteState&>(*state);
    const auto [king, queen] = *belote_state.TrumpMarriage();
    std::vector<std::vector<int>> hands = belote_state.PlayerHands();
    Player king_holder = kInvalidPlayer;
    Player queen_holder = kInvalidPlayer;
    for (Player p = 0; p < kNumPlayers; ++p) {
      if (absl::c_linear_search(hands[p], king)) king_holder = p;
      if (absl::c_linear_search(hands[p], queen)) queen_holder = p;
    }
    SPIEL_CHECK_GE(king_holder, 0);
    SPIEL_CHECK_GE(queen_holder, 0);
    if (king_holder == queen_holder) {
      SPIEL_CHECK_EQ(belote_state.BeloteHolder(), king_holder);
      ++num_held;
    } else {
      SPIEL_CHECK_EQ(belote_state.BeloteHolder(), -1);
      ++num_split;
    }
    // Nothing is announced until one of the two is played.
    SPIEL_CHECK_EQ(belote_state.BeloteAnnounced(), 0);
  }
  SPIEL_CHECK_GT(num_held, 0);
  SPIEL_CHECK_GT(num_split, 0);
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

// Mid-auction the stock still holds 11 cards, any of which could just as well
// be in an opponent's hand, so resampling has to redeal it too.
void ResampleRedealsTheStockDuringTheAuctionTest() {
  std::shared_ptr<const Game> game = LoadGame("belote");
  std::mt19937 rng(24680);
  UniformProbabilitySampler sampler;
  const Player observer = 0;
  bool stock_card_reached_a_hand = false;
  for (int sim = 0; sim < 50; ++sim) {
    std::unique_ptr<State> state = game->NewInitialState();
    while (state->IsChanceNode()) {
      state->ApplyAction(SampleAction(state->ChanceOutcomes(), rng).first);
    }
    const auto& belote_state = static_cast<const BeloteState&>(*state);
    SPIEL_CHECK_EQ(belote_state.CurrentPhase(), Phase::kBid1);
    std::vector<std::vector<int>> hands = belote_state.PlayerHands();

    // The stock is whatever is neither in a hand nor the turned card.
    std::array<bool, kNumCards> in_hand{};
    for (const std::vector<int>& hand : hands) {
      for (int c : hand) in_hand[c] = true;
    }
    in_hand[*belote_state.Upcard()] = true;

    for (int rep = 0; rep < 10; ++rep) {
      std::unique_ptr<State> resampled =
          state->ResampleFromInfostate(observer, sampler);
      const auto& clone = static_cast<const BeloteState&>(*resampled);
      SPIEL_CHECK_EQ(clone.InformationStateString(observer),
                     belote_state.InformationStateString(observer));
      std::vector<std::vector<int>> clone_hands = clone.PlayerHands();
      std::array<int, kNumCards> seen{};
      for (const std::vector<int>& hand : clone_hands) {
        SPIEL_CHECK_EQ(hand.size(), 5);
        for (int c : hand) ++seen[c];
      }
      ++seen[*clone.Upcard()];
      // No card is dealt twice, and 21 of the 32 are accounted for.
      int num_seen = 0;
      for (int c = 0; c < kNumCards; ++c) {
        SPIEL_CHECK_LE(seen[c], 1);
        num_seen += seen[c];
      }
      SPIEL_CHECK_EQ(num_seen, kNumPlayers * 5 + 1);
      for (Player p = 1; p < kNumPlayers; ++p) {
        for (int c : clone_hands[p]) {
          if (!in_hand[c]) stock_card_reached_a_hand = true;
        }
      }
    }
  }
  // If a card that began in the stock never reaches another hand over 500
  // redeals, the stock is not being resampled at all.
  SPIEL_CHECK_TRUE(stock_card_reached_a_hand);
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

// The suit- and trump-following obligations. Each rule is checked wherever it
// comes up across many random deals, and each must come up at least once.
void FollowingObligationsTest() {
  int num_follow = 0;
  int num_forced_ruff = 0;
  int num_partner_winning = 0;
  int num_overtrump = 0;
  ForEachRandomPlayState(800, 11235, [&](const BeloteState& state) {
    std::vector<std::pair<Player, int>> trick = state.CurrentTrick();
    if (trick.empty()) return;
    int trump = state.TrumpSuit();
    int led_suit = CardSuit(trick[0].second);
    Player player = state.CurrentPlayer();
    const std::vector<int> hand = state.PlayerHands()[player];
    std::vector<Action> legal = state.LegalActions();

    Player winner = trick[0].first;
    int best = trick[0].second;
    for (const auto& [p, c] : trick) {
      if (Beats(c, best, led_suit, trump)) {
        best = c;
        winner = p;
      }
    }
    bool partner_winning = PartnerOf(player) == winner;
    bool has_led = absl::c_any_of(
        hand, [&](int c) { return CardSuit(c) == led_suit; });
    bool has_trump = absl::c_any_of(
        hand, [&](int c) { return CardSuit(c) == trump; });

    if (has_led && led_suit != trump) {
      // Holding the led suit in a plain-suit trick: must follow it.
      for (Action a : legal) SPIEL_CHECK_EQ(CardSuit(a), led_suit);
      ++num_follow;
      return;
    }
    if (has_led && led_suit == trump) {
      // Trump was led: must beat the best trump so far if able, even when
      // the partner is winning.
      bool can_beat = absl::c_any_of(hand, [&](int c) {
        return CardSuit(c) == trump &&
               CardStrength(c, trump) > CardStrength(best, trump);
      });
      for (Action a : legal) {
        SPIEL_CHECK_EQ(CardSuit(a), trump);
        if (can_beat) {
          SPIEL_CHECK_GT(CardStrength(a, trump), CardStrength(best, trump));
        }
      }
      if (can_beat) ++num_overtrump;
      return;
    }
    if (!has_trump) return;  // Void in both: free discard, nothing to check.
    if (partner_winning) {
      // The partner holds the trick, so there is no obligation to ruff: every
      // card in hand stays legal.
      SPIEL_CHECK_EQ(legal.size(), hand.size());
      ++num_partner_winning;
    } else {
      // An opponent holds the trick and the player is void: must ruff.
      for (Action a : legal) SPIEL_CHECK_EQ(CardSuit(a), trump);
      ++num_forced_ruff;
    }
  });
  SPIEL_CHECK_GT(num_follow, 0);
  SPIEL_CHECK_GT(num_forced_ruff, 0);
  SPIEL_CHECK_GT(num_partner_winning, 0);
  SPIEL_CHECK_GT(num_overtrump, 0);
}

// A player who is void, is not covered by their partner, and cannot beat the
// opponent's trump must still play a lower trump rather than discard, even
// when holding one. See the note in belote.h.
void MustUndertrumpWhenUnableToOvertrumpTest() {
  int num_checked = 0;
  ForEachRandomPlayState(500, 13579, [&](const BeloteState& state) {
    std::vector<std::pair<Player, int>> trick = state.CurrentTrick();
    if (trick.empty()) return;
    int trump = state.TrumpSuit();
    int led_suit = CardSuit(trick[0].second);
    if (led_suit == trump) return;
    Player player = state.CurrentPlayer();
    const std::vector<int> hand = state.PlayerHands()[player];
    // Void in the led suit, holding at least one trump and one off-suit
    // card that is not trump -- so a discard would be available if the
    // rules allowed one.
    bool has_led = absl::c_any_of(
        hand, [&](int c) { return CardSuit(c) == led_suit; });
    bool has_trump = absl::c_any_of(
        hand, [&](int c) { return CardSuit(c) == trump; });
    bool has_discard = absl::c_any_of(
        hand, [&](int c) { return CardSuit(c) != trump; });
    if (has_led || !has_trump || !has_discard) return;
    // An opponent must currently hold the trick with a trump the player
    // cannot beat.
    Player winner = trick[0].first;
    int best = trick[0].second;
    for (const auto& [p, c] : trick) {
      if (Beats(c, best, led_suit, trump)) {
        best = c;
        winner = p;
      }
    }
    if (PartnerOf(player) == winner) return;
    if (CardSuit(best) != trump) return;
    bool can_overtrump = absl::c_any_of(hand, [&](int c) {
      return CardSuit(c) == trump &&
             CardStrength(c, trump) > CardStrength(best, trump);
    });
    if (can_overtrump) return;
    // Every legal action must be a trump: undertrumping is forced.
    for (Action action : state.LegalActions()) {
      SPIEL_CHECK_EQ(CardSuit(action), trump);
    }
    ++num_checked;
  });
  // The situation is rare but must actually have come up.
  SPIEL_CHECK_GT(num_checked, 0);
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
  open_spiel::belote::CardValuesMatchTheRulesTest();
  open_spiel::belote::ScoreDealTest();
  open_spiel::belote::TerminalScoringTest();
  open_spiel::belote::TakingCompletesTheDealTest();
  open_spiel::belote::BiddingRoundAccessorTest();
  open_spiel::belote::BeloteNeedsASingleHolderTest();
  open_spiel::belote::ManyRandomGamesInvariantsTest();
  open_spiel::belote::AllPassEndsTheGameAsADrawTest();
  open_spiel::belote::BidRoundsAreDistinguishableInformationStatesTest();
  open_spiel::belote::AuctionPassesAreRecordedInBidOrderTest();
  open_spiel::belote::ResampleFromInfostateTest();
  open_spiel::belote::FollowingObligationsTest();
  open_spiel::belote::MustUndertrumpWhenUnableToOvertrumpTest();
  open_spiel::belote::ResampleRedealsTheStockDuringTheAuctionTest();
  open_spiel::belote::BeloteAnnouncementIsPublicTest();
  open_spiel::belote::ResampleFollowsBeloteAnnouncementTest();
}

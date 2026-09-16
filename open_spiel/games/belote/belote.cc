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

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace belote {
namespace {

const GameType kGameType{
    /*short_name=*/"belote",
    /*long_name=*/"Belote",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/kNumPlayers,
    /*min_num_players=*/kNumPlayers,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {
        {"dealer", GameParameter(0)},
    }};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new BeloteGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

open_spiel::RegisterSingleTensorObserver single_tensor(kGameType.short_name);

// Card strength, low to high, when the card's suit is NOT trump. Indexed by
// rank (0=7, 1=8, 2=9, 3=10, 4=J, 5=Q, 6=K, 7=A).
constexpr int kNonTrumpStrength[kNumRanks] = {0, 1, 2, 6, 3, 4, 5, 7};
// Card strength, low to high, when the card's suit IS trump.
constexpr int kTrumpStrength[kNumRanks] = {0, 1, 6, 4, 7, 2, 3, 5};

constexpr int kNonTrumpPoints[kNumRanks] = {0, 0, 0, 10, 2, 3, 4, 11};
constexpr int kTrumpPoints[kNumRanks] = {0, 0, 14, 10, 20, 3, 4, 11};

// A hand holds at most kNumRanks (8) cards, so a fixed-capacity stack array
// avoids heap allocations in the LegalCardPlays hot path.
struct SmallCardList {
  std::array<int, kNumRanks> cards;
  int count = 0;
  void push_back(int card) { cards[count++] = card; }
  bool empty() const { return count == 0; }
  int* begin() { return cards.data(); }
  int* end() { return cards.data() + count; }
  const int* begin() const { return cards.data(); }
  const int* end() const { return cards.data() + count; }
};

std::array<Player, kNumPlayers> OrderFrom(Player start) {
  std::array<Player, kNumPlayers> order{};
  for (int i = 0; i < kNumPlayers; ++i) {
    order[i] = (start + i) % kNumPlayers;
  }
  return order;
}

// Deal order for the first 5 cards/player (3 then 2) plus the turned card.
// A destination of kInvalidPlayer means "turn the next stock card face up".
DealSchedule InitialDealSchedule(Player dealer) {
  std::array<Player, kNumPlayers> order = OrderFrom((dealer + 1) % kNumPlayers);
  DealSchedule schedule;
  for (Player player : order) {
    for (int i = 0; i < 3; ++i) schedule.push_back(player);
  }
  for (Player player : order) {
    for (int i = 0; i < 2; ++i) schedule.push_back(player);
  }
  schedule.push_back(kInvalidPlayer);
  return schedule;
}

// Fisher-Yates shuffle driven by `rng`, a zero-argument callable returning a
// uniform double in [0, 1) -- same convention as ResampleFromInfostate's rng
// parameter, so every shuffle below respects the caller's RNG/seed.
void ShuffleInPlace(std::vector<int>& items,
                    const std::function<double()>& rng) {
  for (int i = static_cast<int>(items.size()) - 1; i > 0; --i) {
    int j = static_cast<int>(rng() * (i + 1));
    std::swap(items[i], items[j]);
  }
}

// Partitions `unseen_cards` among `players` (matching `hand_sizes`) such
// that `allowed(p, c)` holds for every card `c` assigned to player `p`.
// This is always possible here because the constraints are derived from a
// real deal; implemented as a randomized augmenting-path search that finds
// a valid assignment in expected polynomial time. Mirrors belote.py's
// `_bipartite_assign`.
Hands BipartiteAssign(const std::vector<int>& unseen_cards,
                      const std::vector<Player>& players,
                      const std::array<int, kNumPlayers>& hand_sizes,
                      const std::function<bool(Player, int)>& allowed,
                      const std::function<double()>& rng) {
  Hands assigned;
  std::vector<int> player_order(players.begin(), players.end());
  ShuffleInPlace(player_order, rng);

  // Attempts to place `card` with one of the players, possibly reassigning
  // other cards to make room (the augmenting-path step).
  std::function<bool(int, std::array<bool, kNumPlayers>&)> try_place =
      [&](int card, std::array<bool, kNumPlayers>& visited) -> bool {
    for (int p : player_order) {
      if (visited[p] || !allowed(p, card)) continue;
      visited[p] = true;
      if (static_cast<int>(assigned[p].size()) < hand_sizes[p]) {
        assigned[p].push_back(card);
        return true;
      }
      // No room: try to free up a slot by re-homing one of `p`'s cards.
      std::vector<int> bump_order(assigned[p].begin(), assigned[p].end());
      ShuffleInPlace(bump_order, rng);
      for (int other : bump_order) {
        assigned[p].erase(
            std::find(assigned[p].begin(), assigned[p].end(), other));
        if (try_place(other, visited)) {
          assigned[p].push_back(card);
          return true;
        }
        assigned[p].push_back(other);
      }
    }
    return false;
  };

  std::vector<int> cards = unseen_cards;
  ShuffleInPlace(cards, rng);
  for (int card : cards) {
    std::array<bool, kNumPlayers> visited{};
    try_place(card, visited);
  }
  return assigned;
}

}  // namespace

std::string CardString(int card) {
  return absl::StrCat(kRankNames[CardRank(card)],
                      std::string(1, kSuitChar[CardSuit(card)]));
}

int CardPoints(int card, int trump_suit) {
  int rank = CardRank(card);
  return CardSuit(card) == trump_suit ? kTrumpPoints[rank]
                                      : kNonTrumpPoints[rank];
}

int CardStrength(int card, int trump_suit) {
  int rank = CardRank(card);
  return CardSuit(card) == trump_suit ? kTrumpStrength[rank]
                                      : kNonTrumpStrength[rank];
}

BeloteGame::BeloteGame(const GameParameters& params)
    : Game(kGameType, params),
      dealer_(ParameterValue<int>("dealer")) {
  SPIEL_CHECK_GE(dealer_, 0);
  SPIEL_CHECK_LT(dealer_, kNumPlayers);
}

std::vector<int> BeloteGame::InformationStateTensorShape() const {
  // player(4) + hand(32) + dealer(4) + turned_card(32) + trump_suit(5) +
  // declarer(4) + bid_round(3) + current_trick(4 * 32) + cards_played(32) +
  // team_points(2) + belote_announcer(4) + bid1_passes(4) + bid2_passes(4) +
  // trick_history(8 * 4 * 32) + trick_winners(8 * 4).
  int num_tricks = kNumCards / kNumPlayers;
  return {4 + kNumCards + 4 + kNumCards + (kNumSuits + 1) + 4 + 3 +
          kNumPlayers * kNumCards + kNumCards + 2 + kNumPlayers + 4 + 4 +
          num_tricks * kNumPlayers * kNumCards + num_tricks * kNumPlayers};
}

std::vector<int> BeloteGame::ObservationTensorShape() const {
  // Same as the information state tensor, without the per-round pass
  // history, trick history, or trick winners.
  return {4 + kNumCards + 4 + kNumCards + (kNumSuits + 1) + 4 + 3 +
          kNumPlayers * kNumCards + kNumCards + 2 + kNumPlayers};
}

BeloteState::BeloteState(std::shared_ptr<const Game> game, Player dealer)
    : State(game),
      dealer_(dealer),
      deal_schedule_(InitialDealSchedule(dealer)),
      bid_turn_order_(OrderFrom((dealer + 1) % kNumPlayers)) {
  in_deck_.fill(true);
  deck_size_ = kNumCards;
}

Player BeloteState::CurrentPlayer() const {
  if (IsTerminal()) return kTerminalPlayerId;
  if (phase_ == Phase::kDeal) return kChancePlayerId;
  if (phase_ == Phase::kBid1 || phase_ == Phase::kBid2) {
    return bid_turn_order_[bid_pointer_];
  }
  return current_player_play_;
}

std::vector<Action> BeloteState::LegalActions() const {
  if (IsTerminal()) return {};
  if (phase_ == Phase::kDeal) {
    // Cards 0..31 are scanned in index order, which is already ascending, so
    // no sort is needed (unlike sorting a freshly-collected vector).
    std::vector<Action> actions;
    actions.reserve(deck_size_);
    for (int card = 0; card < kNumCards; ++card) {
      if (in_deck_[card]) actions.push_back(card);
    }
    return actions;
  }
  if (phase_ == Phase::kBid1) {
    return {kPassAction, kTakeAction};
  }
  if (phase_ == Phase::kBid2) {
    int turned_suit = CardSuit(turned_card_);
    std::vector<Action> actions = {kPassAction};
    for (int suit = 0; suit < kNumSuits; ++suit) {
      if (suit != turned_suit) actions.push_back(kChooseSuitActionBase + suit);
    }
    return actions;
  }
  SPIEL_CHECK_TRUE(phase_ == Phase::kPlay);
  return LegalCardPlays(current_player_play_);
}

std::vector<Action> BeloteState::LegalCardPlays(Player player) const {
  const auto& hand = hands_[player];
  if (trick_.empty()) {
    // No cards played for the trick, any card may be led.
    std::vector<Action> actions(hand.begin(), hand.end());
    absl::c_sort(actions);
    return actions;
  }

  int led_suit = CardSuit(trick_[0].second);
  int trump = trump_suit_;
  SmallCardList same_suit_cards;
  for (int c : hand) {
    if (CardSuit(c) == led_suit) same_suit_cards.push_back(c);
  }
  Player current_winner = TrickWinner(trick_);
  bool partner_winning = PartnerOf(player) == current_winner;

  if (!same_suit_cards.empty()) {
    if (led_suit != trump) {
      // If the led suit is not trump, must follow suit.
      std::sort(same_suit_cards.begin(), same_suit_cards.end());
      return std::vector<Action>(same_suit_cards.begin(),
                                 same_suit_cards.end());
    }
    // Trump was led: must play higher than the best trump so far if
    // possible, even if the partner currently holds the trick.
    int highest = -1;
    for (const auto& [p, c] : trick_) {
      if (CardSuit(c) == trump) {
        highest = std::max(highest, CardStrength(c, trump));
      }
    }
    SmallCardList higher;
    for (int c : same_suit_cards) {
      if (CardStrength(c, trump) > highest) higher.push_back(c);
    }
    SmallCardList& result = higher.empty() ? same_suit_cards : higher;
    std::sort(result.begin(), result.end());
    return std::vector<Action>(result.begin(), result.end());
  }

  // No cards of the led suit: may play trump if possible.
  SmallCardList trump_cards;
  for (int c : hand) {
    if (CardSuit(c) == trump) trump_cards.push_back(c);
  }
  if (!trump_cards.empty() && led_suit != trump) {
    if (partner_winning) {
      // If the partner is currently winning, any card may be played.
      std::vector<Action> actions(hand.begin(), hand.end());
      absl::c_sort(actions);
      return actions;
    }

    SmallCardList trumps_played;
    for (const auto& [p, c] : trick_) {
      if (CardSuit(c) == trump) trumps_played.push_back(c);
    }
    if (trumps_played.empty()) {
      // No trumps have been played yet, play any trump.
      std::sort(trump_cards.begin(), trump_cards.end());
      return std::vector<Action>(trump_cards.begin(), trump_cards.end());
    }

    // Need to play a higher trump if possible.
    int highest = -1;
    for (int c : trumps_played) {
      highest = std::max(highest, CardStrength(c, trump));
    }
    SmallCardList higher;
    for (int c : trump_cards) {
      if (CardStrength(c, trump) > highest) higher.push_back(c);
    }
    SmallCardList& result = higher.empty() ? trump_cards : higher;
    std::sort(result.begin(), result.end());
    return std::vector<Action>(result.begin(), result.end());
  }

  // No cards of the led suit and no trumps: may play any card.
  std::vector<Action> actions(hand.begin(), hand.end());
  absl::c_sort(actions);
  return actions;
}

bool Beats(int card, int other, int led_suit, int trump) {
  bool card_trump = CardSuit(card) == trump;
  bool other_trump = CardSuit(other) == trump;

  // Exactly one card is trump, so `card` wins iff it is the trump card.
  if (card_trump != other_trump) return card_trump;

  // Both cards are trump, compare by trump ranking order.
  if (card_trump && other_trump) {
    return CardStrength(card, trump) > CardStrength(other, trump);
  }

  bool card_led = CardSuit(card) == led_suit;
  bool other_led = CardSuit(other) == led_suit;

  // Exactly one card follows the led suit, so `card` wins iff it follows the
  // led suit.
  if (card_led != other_led) return card_led;

  // Both cards follow the led suit, compare by non-trump ranking order.
  if (card_led && other_led) {
    return CardStrength(card, trump) > CardStrength(other, trump);
  }

  // Neither card is trump nor led suit: card cannot beat other.
  return false;
}

// ---- public read-only view of the state -----------------------------------

namespace {
// Shared by ToString() and by PhaseString(), which the bindings expose as
// current_phase().
std::string PhaseToString(Phase phase) {
  switch (phase) {
    case Phase::kDeal: return "deal";
    case Phase::kBid1: return "bid1";
    case Phase::kBid2: return "bid2";
    case Phase::kPlay: return "play";
    case Phase::kGameOver: return "done";
  }
  return "";
}
}  // namespace

std::string BeloteState::PhaseString() const { return PhaseToString(phase_); }

absl::optional<int> BeloteState::Upcard() const {
  if (turned_card_ == kInvalidAction) return absl::nullopt;
  return turned_card_;
}

absl::optional<int> BeloteState::BiddingRound() const {
  if (taker_ >= 0) {
    return bid1_passes_.size() == kNumPlayers ? 2 : 1;
  }
  if (phase_ == Phase::kBid1) return 1;
  if (phase_ == Phase::kBid2) return 2;
  return absl::nullopt;
}

std::vector<Player> BeloteState::BidPasses(int round_number) const {
  if (round_number == 1) {
    return std::vector<Player>(bid1_passes_.begin(), bid1_passes_.end());
  }
  if (round_number == 2) {
    return std::vector<Player>(bid2_passes_.begin(), bid2_passes_.end());
  }
  SpielFatalError(
      absl::StrCat("bidding has rounds 1 and 2, not ", round_number));
}

std::vector<std::pair<Player, int>> BeloteState::CurrentTrick() const {
  return std::vector<std::pair<Player, int>>(trick_.begin(), trick_.end());
}

std::vector<Player> BeloteState::TrickWinners() const {
  return std::vector<Player>(trick_winners_.begin(),
                             trick_winners_.begin() + tricks_played_);
}

std::vector<int> BeloteState::PlayedCards() const {
  return std::vector<int>(played_cards_.begin(), played_cards_.end());
}

std::vector<int> BeloteState::TeamPoints() const {
  return std::vector<int>(team_points_.begin(), team_points_.end());
}

std::vector<std::vector<int>> BeloteState::PlayerHands() const {
  std::vector<std::vector<int>> hands;
  hands.reserve(kNumPlayers);
  for (Player player = 0; player < kNumPlayers; ++player) {
    hands.emplace_back(hands_[player].begin(), hands_[player].end());
  }
  return hands;
}

int BeloteState::BeloteAnnounced() const {
  if (belote_holder_ == kInvalidPlayer || trump_suit_ < 0) return 0;
  const auto [trump_king, trump_queen] = TrumpKingAndQueen();
  int announced = 0;
  for (int card : played_cards_) {
    if (card == trump_king || card == trump_queen) ++announced;
  }
  return announced;
}

std::vector<std::vector<std::pair<Player, int>>> BeloteState::Tricks() const {
  std::vector<std::vector<std::pair<Player, int>>> result;
  for (const Trick& trick : ReconstructTricks()) {
    result.emplace_back(trick.begin(), trick.end());
  }
  return result;
}

absl::optional<std::pair<int, int>> BeloteState::TrumpMarriage() const {
  if (trump_suit_ < 0) return absl::nullopt;
  return TrumpKingAndQueen();
}

VoidAndTrumpBounds BeloteState::PublicInference() const {
  return InferVoidAndTrumpBounds(ReconstructTricks());
}

bool BeloteState::IsBetter(int card, int other, int led_suit) const {
  return Beats(card, other, led_suit, trump_suit_);
}

Player BeloteState::TrickWinner(const Trick& trick) const {
  int led_suit = CardSuit(trick[0].second);
  Player best_player = trick[0].first;
  int best_card = trick[0].second;
  for (int i = 1; i < trick.size(); ++i) {
    if (IsBetter(trick[i].second, best_card, led_suit)) {
      best_player = trick[i].first;
      best_card = trick[i].second;
    }
  }
  return best_player;
}

std::vector<std::pair<Action, double>> BeloteState::ChanceOutcomes() const {
  SPIEL_CHECK_TRUE(phase_ == Phase::kDeal);
  double probability = 1.0 / deck_size_;
  std::vector<std::pair<Action, double>> outcomes;
  outcomes.reserve(deck_size_);
  for (int card = 0; card < kNumCards; ++card) {
    if (in_deck_[card]) outcomes.emplace_back(card, probability);
  }
  return outcomes;
}

void BeloteState::EnterPlayPhase() {
  trick_leader_ = (dealer_ + 1) % kNumPlayers;
  current_player_play_ = trick_leader_;
  trick_.clear();
  tricks_played_ = 0;
  team_points_ = {0, 0};
  belote_holder_ = FindBeloteHolder();
}

std::pair<int, int> BeloteState::TrumpKingAndQueen() const {
  return {trump_suit_ * kNumRanks + 6,   // Rank index of "K".
          trump_suit_ * kNumRanks + 5};  // Rank index of "Q".
}

Player BeloteState::FindBeloteHolder(const Hands& hands,
                                     const TrickList& tricks) const {
  const auto [trump_king, trump_queen] = TrumpKingAndQueen();
  std::array<absl::InlinedVector<int, kNumRanks>, kNumPlayers> played_by;
  for (const Trick& trick : tricks) {
    for (const auto& [player, card] : trick) played_by[player].push_back(card);
  }
  for (Player player = 0; player < kNumPlayers; ++player) {
    bool has_king = absl::c_linear_search(hands[player], trump_king) ||
                    absl::c_linear_search(played_by[player], trump_king);
    bool has_queen = absl::c_linear_search(hands[player], trump_queen) ||
                     absl::c_linear_search(played_by[player], trump_queen);
    if (has_king && has_queen) return player;
  }
  return kInvalidPlayer;
}

Player BeloteState::FindBeloteHolder() const {
  return FindBeloteHolder(hands_, ReconstructTricks());
}

TrickList BeloteState::ReconstructCompletedTricks() const {
  TrickList tricks;
  Player leader = (dealer_ + 1) % kNumPlayers;
  for (int i = 0; i < tricks_played_; ++i) {
    std::array<Player, kNumPlayers> order = OrderFrom(leader);
    Trick trick;
    for (int j = 0; j < kNumPlayers; ++j) {
      trick.emplace_back(order[j], trick_history_[i][j]);
    }
    tricks.push_back(std::move(trick));
    leader = trick_winners_[i];
  }
  return tricks;
}

TrickList BeloteState::ReconstructTricks() const {
  TrickList tricks = ReconstructCompletedTricks();
  if (!trick_.empty()) tricks.push_back(trick_);
  return tricks;
}

VoidAndTrumpBounds BeloteState::InferVoidAndTrumpBounds(
    const TrickList& tricks) const {
  VoidAndTrumpBounds result;
  int trump = trump_suit_;
  if (trump < 0) return result;
  for (const Trick& trick : tricks) {
    if (trick.empty()) continue;
    int led_suit = CardSuit(trick[0].second);
    for (int idx = 1; idx < static_cast<int>(trick.size()); ++idx) {
      Player player = trick[idx].first;
      int card = trick[idx].second;
      int suit = CardSuit(card);
      Trick partial(trick.begin(), trick.begin() + idx);
      Player current_winner = TrickWinner(partial);
      bool partner_winning = PartnerOf(player) == current_winner;
      SmallCardList trumps_played_before;
      for (const auto& [p, c] : partial) {
        if (CardSuit(c) == trump) trumps_played_before.push_back(c);
      }

      if (suit != led_suit) {
        // Not following was only legal if void in that suit.
        result.void_suits[player][led_suit == trump ? trump : led_suit] =
            true;
        if (suit != trump && led_suit != trump && !partner_winning) {
          // Trumping in was mandatory here too, so void there.
          result.void_suits[player][trump] = true;
        }
      }

      if (suit == trump && !trumps_played_before.empty()) {
        bool forced_to_overtrump = (led_suit == trump) || !partner_winning;
        int highest = -1;
        for (int c : trumps_played_before) {
          highest = std::max(highest, CardStrength(c, trump));
        }
        if (forced_to_overtrump && CardStrength(card, trump) <= highest) {
          // Didn't overtrump though forced to if possible: no trump above
          // `highest` remains in hand.
          int& bound = result.max_trump_strength[player];
          if (bound == -1 || highest < bound) bound = highest;
        }
      }
    }
  }
  return result;
}

Player BeloteState::AnnouncedBeloteHolder() const {
  return BeloteAnnounced() > 0 ? belote_holder_ : kInvalidPlayer;
}

std::pair<Player, int> BeloteState::PublicCardBar() const {
  // With a holder, a lone played marriage card was theirs, and announced.
  if (trump_suit_ < 0 || belote_holder_ >= 0) return {kInvalidPlayer, -1};
  const auto [trump_king, trump_queen] = TrumpKingAndQueen();
  bool king_played = absl::c_linear_search(played_cards_, trump_king);
  bool queen_played = absl::c_linear_search(played_cards_, trump_queen);
  if (king_played == queen_played) return {kInvalidPlayer, -1};
  int played = king_played ? trump_king : trump_queen;
  for (const Trick& trick : ReconstructTricks()) {
    for (const auto& [player, card] : trick) {
      if (card == played) {
        return {player, king_played ? trump_queen : trump_king};
      }
    }
  }
  SpielFatalError("A played card is missing from the tricks.");
}

std::array<absl::InlinedVector<int, 2>, kNumPlayers> BeloteState::PublicCardPins(
    Player player_id) const {
  std::array<absl::InlinedVector<int, 2>, kNumPlayers> pins;
  auto pin = [&](Player player, int card) {
    if (player == player_id) return;
    if (!absl::c_linear_search(hands_[player], card)) return;
    if (!absl::c_linear_search(pins[player], card)) {
      pins[player].push_back(card);
    }
  };
  if (turned_card_ != kInvalidAction && taker_ >= 0) {
    pin(taker_, turned_card_);
  }
  if (belote_holder_ >= 0) {
    int trump_king = trump_suit_ * kNumRanks + 6;
    int trump_queen = trump_suit_ * kNumRanks + 5;
    bool king_played = absl::c_linear_search(played_cards_, trump_king);
    bool queen_played = absl::c_linear_search(played_cards_, trump_queen);
    if (king_played != queen_played) {
      pin(belote_holder_, king_played ? trump_queen : trump_king);
    }
  }
  return pins;
}

void BeloteState::ApplyDealAction(int card) {
  in_deck_[card] = false;
  --deck_size_;
  Player destination = deal_schedule_[deal_index_];
  if (destination == kInvalidPlayer) {
    turned_card_ = card;
  } else {
    hands_[destination].push_back(card);
  }
  ++deal_index_;
  if (deal_index_ == deal_schedule_.size()) {
    phase_ = after_deal_phase_;
    deal_schedule_.clear();
    deal_index_ = 0;
    if (phase_ == Phase::kPlay) EnterPlayPhase();
  }
}

void BeloteState::StartCompletionDeal(DealSchedule schedule,
                                      Phase next_phase) {
  deal_schedule_ = std::move(schedule);
  deal_index_ = 0;
  after_deal_phase_ = next_phase;
  phase_ = Phase::kDeal;
}

DealSchedule BeloteState::CompletionScheduleAfterTake(Player taker) const {
  // 3 cards to each non-taker, 2 to the taker (who already holds the turned
  // card).
  std::array<Player, kNumPlayers> order = OrderFrom((dealer_ + 1) % kNumPlayers);
  std::array<int, kNumPlayers> target_counts{};
  std::array<int, kNumPlayers> dealt_counts{};
  for (Player p : order) target_counts[p] = (p == taker) ? 2 : 3;
  DealSchedule schedule;
  bool remaining = true;
  while (remaining) {
    remaining = false;
    for (Player p : order) {
      if (dealt_counts[p] < target_counts[p]) {
        schedule.push_back(p);
        ++dealt_counts[p];
        if (dealt_counts[p] < target_counts[p]) remaining = true;
      }
    }
  }
  return schedule;
}

void BeloteState::ApplyBid1Action(int action, Player player) {
  if (action == kTakeAction) {
    taker_ = player;
    trump_suit_ = CardSuit(turned_card_);
    declarer_team_ = TeamOf(player);
    hands_[player].push_back(turned_card_);
    StartCompletionDeal(CompletionScheduleAfterTake(player), Phase::kPlay);
  } else {
    bid1_passes_.push_back(player);
    ++bid_pointer_;
    if (bid_pointer_ == kNumPlayers) {
      phase_ = Phase::kBid2;
      bid_pointer_ = 0;
    }
  }
}

void BeloteState::ApplyBid2Action(int action, Player player) {
  if (action == kPassAction) {
    bid2_passes_.push_back(player);
    ++bid_pointer_;
    if (bid_pointer_ == kNumPlayers) {
      // Everyone passed twice: the deal is thrown in and scores nothing.
      phase_ = Phase::kGameOver;
      returns_.fill(0.0);
    }
  } else {
    int suit = action - kChooseSuitActionBase;
    taker_ = player;
    trump_suit_ = suit;
    declarer_team_ = TeamOf(player);
    hands_[player].push_back(turned_card_);
    StartCompletionDeal(CompletionScheduleAfterTake(player), Phase::kPlay);
  }
}

void BeloteState::FinalizeScores() {
  int other_team = 1 - declarer_team_;
  int declarer_points = team_points_[declarer_team_];
  int other_points = team_points_[other_team];
  // 162 normally, or 252 if one team won all 8 tricks (capot).
  int trick_total = declarer_points + other_points;
  int belote_team = (belote_holder_ >= 0) ? TeamOf(belote_holder_) : -1;
  int declarer_bonus =
      (belote_team == declarer_team_) ? kBeloteRebeloteBonus : 0;
  int other_bonus = (belote_team == other_team) ? kBeloteRebeloteBonus : 0;
  int final_declarer, final_other;
  // Contract success/failure is decided on totals that include the
  // belote/rebelote bonus, not on trick points alone.
  if (declarer_points + declarer_bonus > other_points + other_bonus) {
    final_declarer = declarer_points;
    final_other = other_points;
  } else {
    final_declarer = 0;
    final_other = trick_total;
  }
  final_declarer += declarer_bonus;
  final_other += other_bonus;
  double diff = static_cast<double>(final_declarer - final_other);
  for (Player p = 0; p < kNumPlayers; ++p) {
    returns_[p] = (TeamOf(p) == declarer_team_) ? diff : -diff;
  }
}

void BeloteState::ApplyPlayAction(int card, Player player) {
  auto& hand = hands_[player];
  hand.erase(std::remove(hand.begin(), hand.end(), card), hand.end());
  trick_.emplace_back(player, card);
  played_cards_.push_back(card);
  if (trick_.size() < kNumPlayers) {
    current_player_play_ = (player + 1) % kNumPlayers;
    return;
  }

  Player winner = TrickWinner(trick_);
  int points = 0;
  for (const auto& [p, c] : trick_) points += CardPoints(c, trump_suit_);
  ++tricks_played_;
  int trick_index = tricks_played_ - 1;
  trick_winners_[trick_index] = winner;
  if (tricks_played_ == kNumTricks) {
    bool is_capot = std::all_of(
        trick_winners_.begin(), trick_winners_.begin() + tricks_played_,
        [winner](Player w) { return TeamOf(w) == TeamOf(winner); });
    points += is_capot ? kCapotLastTrickBonus : kLastTrickBonus;
  }
  team_points_[TeamOf(winner)] += points;
  for (int j = 0; j < kNumPlayers; ++j) {
    trick_history_[trick_index][j] = trick_[j].second;
  }

  trick_.clear();
  trick_leader_ = winner;
  current_player_play_ = winner;
  if (tricks_played_ == kNumCards / kNumPlayers) {
    FinalizeScores();
    phase_ = Phase::kGameOver;
  }
}

void BeloteState::DoApplyAction(Action action) {
  switch (phase_) {
    case Phase::kDeal:
      return ApplyDealAction(action);
    case Phase::kBid1:
      return ApplyBid1Action(action, bid_turn_order_[bid_pointer_]);
    case Phase::kBid2:
      return ApplyBid2Action(action, bid_turn_order_[bid_pointer_]);
    case Phase::kPlay:
      return ApplyPlayAction(action, current_player_play_);
    case Phase::kGameOver:
      SpielFatalError("Cannot act in terminal states");
  }
}

std::string BeloteState::ActionToString(Player player, Action action) const {
  if (player == kChancePlayerId) return absl::StrCat("Deal: ", CardString(action));
  if (action == kPassAction) return "Pass";
  if (action == kTakeAction) return "Take";
  if (action >= kChooseSuitActionBase &&
      action < kChooseSuitActionBase + kNumSuits) {
    return absl::StrCat("Choose trump: ",
                        std::string(1, kSuitChar[action - kChooseSuitActionBase]));
  }
  return absl::StrCat("Play: ", CardString(action));
}

namespace {
// Cards as a quoted list, e.g. ['10C', 'KC']. The quoting is part of the
// observation-string format that tests pin, so it must not drift.
template <typename Container>
std::string CardListString(const Container& cards) {
  std::string rv = "[";
  bool first = true;
  for (int card : cards) {
    if (!first) absl::StrAppend(&rv, ", ");
    first = false;
    absl::StrAppend(&rv, "'", CardString(card), "'");
  }
  absl::StrAppend(&rv, "]");
  return rv;
}

template <typename Container>
std::string HandString(const Container& hand) {
  std::vector<int> sorted_hand(hand.begin(), hand.end());
  absl::c_sort(sorted_hand);
  return absl::StrCat("[", absl::StrJoin(sorted_hand, ", "), "]");
}
}  // namespace

std::string BeloteState::ToString() const {
  std::string rv;
  absl::StrAppend(&rv, "Dealer: ", dealer_, "\n");
  absl::StrAppend(&rv, "Phase: ", PhaseToString(phase_), "\n");
  absl::StrAppend(&rv, "Hands: [");
  for (int p = 0; p < kNumPlayers; ++p) {
    if (p > 0) absl::StrAppend(&rv, ", ");
    absl::StrAppend(&rv, HandString(hands_[p]));
  }
  absl::StrAppend(&rv, "]\n");
  if (turned_card_ != kInvalidAction) {
    absl::StrAppend(&rv, "Turned card: ", CardString(turned_card_), "\n");
  }
  if (trump_suit_ >= 0) {
    absl::StrAppend(&rv, "Trump: ", std::string(1, kSuitChar[trump_suit_]),
                    ", Taker: ", taker_, "\n");
  }
  if (phase_ == Phase::kPlay || phase_ == Phase::kGameOver) {
    absl::StrAppend(&rv, "Trick: [");
    for (int i = 0; i < trick_.size(); ++i) {
      if (i > 0) absl::StrAppend(&rv, ", ");
      absl::StrAppend(&rv, "(", trick_[i].first, ", ",
                      CardString(trick_[i].second), ")");
    }
    absl::StrAppend(&rv, "]\n");
    absl::StrAppend(&rv, "Team points: [", team_points_[0], ", ",
                    team_points_[1], "]\n");
    if (belote_holder_ >= 0) {
      absl::StrAppend(&rv, "Belote/rebelote holder: ", belote_holder_,
                      " (team ", TeamOf(belote_holder_), ")\n");
    }
  }
  return rv;
}

void BeloteState::WriteObservation(Player player, bool perfect_recall,
                                   absl::Span<float> values) const {
  std::fill(values.begin(), values.end(), 0.0f);
  auto it = values.begin();
  it[player] = 1;
  it += kNumPlayers;
  for (int card : hands_[player]) it[card] = 1;
  it += kNumCards;
  it[dealer_] = 1;
  it += kNumPlayers;
  if (turned_card_ != kInvalidAction) it[turned_card_] = 1;
  it += kNumCards;
  it[trump_suit_ >= 0 ? trump_suit_ : kNumSuits] = 1;
  it += (kNumSuits + 1);
  if (taker_ >= 0) it[taker_] = 1;
  it += kNumPlayers;

  // Which bidding round is in progress (none / round 1 / round 2). A
  // present-tense fact about the current state, so it belongs in the plain
  // observation too, not only the perfect-recall one: without it a bid1
  // state and the bid2 state that follows are indistinguishable (identical
  // hand, identical turned card, trump still unset) even though they offer
  // different actions.
  it[phase_ == Phase::kBid1 ? 1 : (phase_ == Phase::kBid2 ? 2 : 0)] = 1;
  it += 3;

  // One card-slot per player (indexed by absolute player id) rather than a
  // single unordered bag, so which player played which card -- and hence
  // the led suit and who currently holds the trick -- can be recovered from
  // the tensor. Same convention as the C++ Euchre implementation's trick
  // encoding. `trick_` is already exactly these (player, card) pairs in
  // play order, so no reconstruction is needed here.
  for (const auto& [p, c] : trick_) {
    it[p * kNumCards + c] = 1;
  }
  it += kNumPlayers * kNumCards;

  for (int card : played_cards_) it[card] = 1;
  it += kNumCards;
  it[0] = team_points_[0] / static_cast<float>(kMaxScoreCapot);
  it[1] = team_points_[1] / static_cast<float>(kMaxScoreCapot);
  it += 2;
  // Who announced belote, if anyone has. Public the moment the holder plays
  // the first of the trump King and Queen, and stays so.
  Player belote_announcer = AnnouncedBeloteHolder();
  if (belote_announcer >= 0) it[belote_announcer] = 1;
  it += kNumPlayers;
  if (perfect_recall) {
    // Who passed, per round, indexed by absolute player id. History rather
    // than present state, hence perfect-recall only -- and public history:
    // everyone hears every pass. This is what lets a policy condition on
    // "three players already declined this suit" instead of seeing only its
    // own cards.
    for (Player p : bid1_passes_) it[p] = 1;
    it += kNumPlayers;
    for (Player p : bid2_passes_) it[p] = 1;
    it += kNumPlayers;
    // Only reconstructed for InformationStateTensor: ObservationTensor has
    // no use for the completed-trick history, so it never pays this cost.
    TrickList tricks = ReconstructCompletedTricks();
    for (int trick_idx = 0; trick_idx < tricks_played_; ++trick_idx) {
      for (const auto& [p, c] : tricks[trick_idx]) {
        it[(trick_idx * kNumPlayers + p) * kNumCards + c] = 1;
      }
    }
    it += kNumTricks * kNumPlayers * kNumCards;
    // Winner of each completed trick; combined with `dealer` (who leads
    // trick 0) this lets a consumer chain trick leaders forward (the winner
    // of trick i leads trick i+1).
    for (int trick_idx = 0; trick_idx < tricks_played_; ++trick_idx) {
      it[trick_idx * kNumPlayers + trick_winners_[trick_idx]] = 1;
    }
  }
}

void BeloteState::InformationStateTensor(Player player,
                                         absl::Span<float> values) const {
  WriteObservation(player, /*perfect_recall=*/true, values);
}

void BeloteState::ObservationTensor(Player player,
                                    absl::Span<float> values) const {
  WriteObservation(player, /*perfect_recall=*/false, values);
}

// Who passed, and the completed-trick history, are history rather than
// present state, so they belong only in the perfect-recall string: the plain
// observation must not carry them, and the observation tensor does not.
std::string BeloteState::ObservationStringImpl(Player player,
                                               bool perfect_recall) const {
  std::string rv;
  absl::StrAppend(&rv, "p", player);
  absl::InlinedVector<int, kNumRanks> hand(hands_[player].begin(),
                                           hands_[player].end());
  absl::c_sort(hand);
  absl::StrAppend(&rv, " hand:", CardListString(hand));
  absl::StrAppend(&rv, " dealer:", dealer_);
  if (turned_card_ != kInvalidAction) {
    absl::StrAppend(&rv, " turned:", CardString(turned_card_));
  }
  if (trump_suit_ >= 0) {
    absl::StrAppend(&rv, " trump:", std::string(1, kSuitChar[trump_suit_]));
  }
  if (taker_ >= 0) absl::StrAppend(&rv, " declarer:", taker_);
  Player belote_announcer = AnnouncedBeloteHolder();
  if (belote_announcer >= 0) {
    absl::StrAppend(&rv, " belote:", belote_announcer);
  }
  if (phase_ == Phase::kBid1 || phase_ == Phase::kBid2) {
    absl::StrAppend(&rv, " bidround:", phase_ == Phase::kBid1 ? "1" : "2");
  }
  if (perfect_recall) {
    if (!bid1_passes_.empty()) {
      absl::StrAppend(&rv, " passed1:[", absl::StrJoin(bid1_passes_, ", "),
                      "]");
    }
    if (!bid2_passes_.empty()) {
      absl::StrAppend(&rv, " passed2:[", absl::StrJoin(bid2_passes_, ", "),
                      "]");
    }
  }
  absl::InlinedVector<int, kNumPlayers> trick_cards;
  for (const auto& [unused_player, card] : trick_) trick_cards.push_back(card);
  absl::StrAppend(&rv, " trick:", CardListString(trick_cards));
  absl::StrAppend(&rv, " played:", CardListString(played_cards_));
  if (perfect_recall && tricks_played_ > 0) {
    absl::StrAppend(&rv, " history:");
    for (int i = 0; i < tricks_played_; ++i) {
      if (i > 0) absl::StrAppend(&rv, "|");
      for (int j = 0; j < kNumPlayers; ++j) {
        if (j > 0) absl::StrAppend(&rv, ",");
        absl::StrAppend(&rv, CardString(trick_history_[i][j]));
      }
    }
  }
  absl::StrAppend(&rv, " points:[", team_points_[0], ", ", team_points_[1],
                  "]");
  return rv;
}

std::string BeloteState::InformationStateString(Player player) const {
  return ObservationStringImpl(player, /*perfect_recall=*/true);
}

std::string BeloteState::ObservationString(Player player) const {
  return ObservationStringImpl(player, /*perfect_recall=*/false);
}

// Returns a clone with the other players' hands resampled, kept consistent
// with `player_id`'s information state: own hand and public history
// untouched, cards pinned by `PublicCardPins` kept with their known holder,
// and the rest resampled via `BipartiteAssign` under the void-suit and
// trump-strength constraints from `InferVoidAndTrumpBounds` and the card
// barred by `PublicCardBar`. `rng` is a
// zero-argument callable returning a uniform double in [0, 1), used to
// drive every shuffle so this respects the caller's RNG/seed. Mirrors
// belote.py's `resample_from_infostate`.
std::unique_ptr<State> BeloteState::ResampleFromInfostate(
    int player_id, std::function<double()> rng) const {
  std::unique_ptr<BeloteState> state(new BeloteState(*this));

  TrickList tricks = ReconstructTricks();
  std::array<absl::InlinedVector<int, 2>, kNumPlayers> pinned =
      PublicCardPins(player_id);

  std::vector<Player> other_players;
  std::array<int, kNumPlayers> hand_sizes{};
  std::vector<int> unseen_cards;
  for (Player p = 0; p < kNumPlayers; ++p) {
    if (p == player_id) continue;
    other_players.push_back(p);
    std::vector<int> cards;
    for (int c : hands_[p]) {
      if (!absl::c_linear_search(pinned[p], c)) cards.push_back(c);
    }
    hand_sizes[p] = static_cast<int>(cards.size());
    unseen_cards.insert(unseen_cards.end(), cards.begin(), cards.end());
  }

  VoidAndTrumpBounds bounds = InferVoidAndTrumpBounds(tricks);
  int trump = trump_suit_;
  const auto [barred_player, barred_card] = PublicCardBar();
  std::function<bool(Player, int)> allowed = [&bounds, trump, barred_player,
                                              barred_card](Player p,
                                                           int card) {
    if (p == barred_player && card == barred_card) return false;
    int suit = CardSuit(card);
    if (bounds.void_suits[p][suit]) return false;
    int bound = bounds.max_trump_strength[p];
    return !(suit == trump && bound != -1 && CardStrength(card, trump) > bound);
  };

  Hands assignment =
      BipartiteAssign(unseen_cards, other_players, hand_sizes, allowed, rng);
  for (Player p : other_players) {
    for (int c : pinned[p]) assignment[p].push_back(c);
    state->hands_[p] = std::move(assignment[p]);
  }

  // Check if the resampled hands reveal a belote/rebelote holder.
  state->belote_holder_ = state->FindBeloteHolder(state->hands_, tricks);

  return state;
}

}  // namespace belote
}  // namespace open_spiel

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

#ifndef OPEN_SPIEL_GAMES_BELOTE_H_
#define OPEN_SPIEL_GAMES_BELOTE_H_

// Classic (non-contract) French Belote: 4 players in two fixed partnerships
// (0 & 2 vs 1 & 3), a 32-card deck, 8 tricks.
//
// https://en.wikipedia.org/wiki/Belote
// https://www.pagat.com/jass/belote.html
// https://www.ffbelote.org/regles-officielle-belote/
//
// Trump is chosen by the "prise": 5 cards are dealt to each player and the
// next stock card is turned face up. Players in turn may take it, making its
// suit trump (round 1); if all four pass, they may instead name one of the
// other three suits (round 2). The taker adds the turned card to their hand
// and the deal is completed to 8 cards each.
//
// Scoring uses the 162-point deck (152 in cards plus 10 for the last trick).
// The declaring team keeps its trick points only if it scores strictly more
// than the defenders; otherwise the defenders collect all trick points. A
// capot (one team wins all 8 tricks) raises the last-trick bonus to 100, so
// the deck is worth 252.
//
// Following a trick: you must follow suit; if void you must trump, unless
// your partner is already winning the trick; and whenever you play a trump
// you must beat the highest trump in the trick if you can. A player who is
// void and cannot overtrump must still play a trump even when holding a
// discard -- pagat states this explicitly ("he must still play a trump,
// although he does not benefit from doing so"), though some casual rule sets
// let the player discard instead.
//
// Two deliberate departures from the official rules:
//
//   * Four passes in round 2 end the deal as a draw (all returns 0) instead
//     of triggering a redeal, since a state here is a single deal. Same
//     choice as bridge's passed out hand.
//   * The belote/rebelote bonus (20 points to the team of a player holding
//     both the King and Queen of trump) is always awarded. There is no
//     announcement action: the holder is announced to every player
//     automatically when they play the first of the two cards, so a player
//     who plays one of them without an announcement is publicly known not
//     to hold the other. The 20 points follow the official rules -- they
//     count toward whichever team's contract threshold applies and are
//     credited to the holder's team win or lose.

#include <array>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/inlined_vector.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"

namespace open_spiel {
namespace belote {

inline constexpr int kNumPlayers = 4;
inline constexpr int kNumSuits = 4;
inline constexpr int kNumRanks = 8;
inline constexpr int kNumCards = kNumSuits * kNumRanks;
inline constexpr int kMaxScore = 162;
inline constexpr int kLastTrickBonus = 10;
inline constexpr int kCapotLastTrickBonus = 100;
inline constexpr int kMaxScoreCapot =
    kMaxScore - kLastTrickBonus + kCapotLastTrickBonus;
inline constexpr int kBeloteRebeloteBonus = 20;
inline constexpr int kNumTricks = kNumCards / kNumPlayers;
// Longest possible deal_schedule_: the initial deal (3+2 cards to each of 4
// players, plus the turned card).
inline constexpr int kMaxDealScheduleSize = kNumPlayers * 5 + 1;

// Card actions are 0..31 (card = suit * kNumRanks + rank).
inline constexpr int kPassAction = kNumCards;                         // 32
inline constexpr int kTakeAction = kNumCards + 1;                     // 33
inline constexpr int kChooseSuitActionBase = kNumCards + 2;           // 34..37
inline constexpr int kNumDistinctActions = kNumCards + 2 + kNumSuits;  // 38

constexpr char kSuitChar[] = "CDHS";
// Ranks, low to high face value: 7 8 9 10 J Q K A.
constexpr const char* kRankNames[] = {"7", "8", "9", "10",
                                      "J", "Q", "K", "A"};
// Rank indices of the two cards making up the belote/rebelote marriage.
inline constexpr int kQueenRank = 5;
inline constexpr int kKingRank = 6;

enum class Phase { kDeal, kBid1, kBid2, kPlay, kGameOver };
std::ostream& operator<<(std::ostream& os, const Phase& phase);

inline int CardSuit(int card) { return card / kNumRanks; }
inline int CardRank(int card) { return card % kNumRanks; }
std::string CardString(int card);
int CardPoints(int card, int trump_suit);
int CardStrength(int card, int trump_suit);
// Whether `card` beats `other` within the same trick. Depends only on the two
// cards, the led suit and the trump suit, so a strategy can weigh a
// hypothetical trick without a state to hang the call on.
bool Beats(int card, int other, int led_suit, int trump_suit);

// Which side, if either, holds the belote/rebelote marriage.
enum class BeloteSide { kNone, kDeclarers, kDefenders };

// Final totals for (declaring team, defending team) given their raw trick
// points, applying the all-or-nothing rule and the 20-point belote/rebelote
// bonus. Like Beats, this depends only on its arguments, so a strategy can
// ask "do we still make the contract if we lose this trick?" with no state
// to hang the call on.
std::pair<int, int> ScoreDeal(int declarer_points, int defender_points,
                              BeloteSide belote_side);
inline int TeamOf(Player player) { return player % 2; }
inline Player PartnerOf(Player player) { return (player + 2) % kNumPlayers; }

// A trick in progress or completed, as (player, card) pairs in play order.
using Trick = absl::InlinedVector<std::pair<Player, int>, kNumPlayers>;
// A run of tricks (e.g. every trick played so far, current one included).
using TrickList = absl::InlinedVector<Trick, kNumTricks + 1>;
// Cards held by one player, or any other bounded set of cards from a hand.
using CardList = absl::InlinedVector<int, kNumRanks>;
// One hand of cards per player, indexed by absolute player id.
using Hands = std::array<CardList, kNumPlayers>;
// Which player, or kInvalidPlayer for "turn face up", gets each successive
// card of a deal.
using DealSchedule = absl::InlinedVector<Player, kMaxDealScheduleSize>;

// Void suits and trump-strength upper bounds inferred per player from public
// play, used to constrain opponent hand resampling. `max_trump_strength[p]`
// of -1 means no known bound.
struct VoidAndTrumpBounds {
  std::array<std::array<bool, kNumSuits>, kNumPlayers> void_suits{};
  std::array<int, kNumPlayers> max_trump_strength = {-1, -1, -1, -1};
};

class BeloteState : public State {
 public:
  explicit BeloteState(std::shared_ptr<const Game> game, Player dealer);
  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  bool IsTerminal() const override { return phase_ == Phase::kGameOver; }
  std::vector<double> Returns() const override {
    return std::vector<double>(returns_.begin(), returns_.end());
  }
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override {
    return std::unique_ptr<State>(new BeloteState(*this));
  }
  std::vector<Action> LegalActions() const override;
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::unique_ptr<State> ResampleFromInfostate(
      int player_id, std::function<double()> rng) const override;

  Phase CurrentPhase() const { return phase_; }
  Player Dealer() const { return dealer_; }
  // The card turned face up for round 1. Stays set once the auction resolves.
  absl::optional<int> Upcard() const;
  // The seat holding the contract, or -1. Euchre calls this the declarer.
  Player Taker() const { return taker_ >= 0 ? taker_ : -1; }
  int DeclarerTeam() const { return declarer_team_; }
  int TrumpSuit() const { return trump_suit_; }
  // 1 or 2, or absl::nullopt outside the auction. Unlike CurrentPhase(), this
  // still reports the round that resolved the contract.
  absl::optional<int> BiddingRound() const;
  // Seats that passed in `round_number` (1 or 2), in turn order.
  std::vector<Player> BidPasses(int round_number) const;
  std::vector<std::pair<Player, int>> CurrentTrick() const;
  std::vector<Player> TrickWinners() const;
  std::vector<int> PlayedCards() const;
  // Includes the last-trick bonus, excludes the belote bonus.
  std::vector<int> TeamPoints() const;
  std::vector<std::vector<int>> PlayerHands() const;
  // The seat holding both trump King and Queen, or -1. Private until
  // BeloteAnnounced() is non-zero.
  Player BeloteHolder() const {
    return belote_holder_ >= 0 ? belote_holder_ : -1;
  }
  // How many of the two marriage cards the holder has played (0-2).
  int BeloteAnnounced() const;
  // Every trick as (player, card) pairs, including the one in progress.
  std::vector<std::vector<std::pair<Player, int>>> Tricks() const;
  absl::optional<std::pair<int, int>> TrumpMarriage() const;
  // What the public history proves about the other hands, as used by
  // ResampleFromInfostate.
  VoidAndTrumpBounds PublicInference() const;

 protected:
  void DoApplyAction(Action action) override;

 private:
  std::vector<Action> LegalCardPlays(Player player) const;
  bool IsBetter(int card, int other, int led_suit) const;
  // Only meaningful once a trump suit exists; TrumpMarriage() guards it.
  std::pair<int, int> TrumpKingAndQueen() const;
  Player TrickWinner(const Trick& trick) const;
  void EnterPlayPhase();
  // The player holding, or already seen to have played, both the King and
  // Queen of trump, or kInvalidPlayer. Played cards count, because who
  // played what holds across resampled worlds.
  Player FindBeloteHolder(const Hands& hands, const TrickList& tricks) const;
  Player FindBeloteHolder() const;
  void ApplyDealAction(int card);
  void StartCompletionDeal(DealSchedule schedule, Phase next_phase);
  DealSchedule CompletionScheduleAfterTake(Player taker) const;
  void ApplyBid1Action(int action, Player player);
  void ApplyBid2Action(int action, Player player);
  void ApplyPlayAction(int card, Player player);
  void FinalizeScores();
  void WriteObservation(Player player, bool perfect_recall,
                        absl::Span<float> values) const;
  // Backs both InformationStateString and ObservationString; `perfect_recall`
  // gates the history, as in WriteObservation.
  std::string ObservationStringImpl(Player player, bool perfect_recall) const;
  // Trick 0 is led by the player after the dealer; trick i>0 by the winner
  // of trick i-1.
  TrickList ReconstructCompletedTricks() const;
  // As above, plus the current partial trick (if any) appended at the end.
  TrickList ReconstructTricks() const;
  VoidAndTrumpBounds InferVoidAndTrumpBounds(const TrickList& tricks) const;
  // Cards whose holder is public knowledge: the turned card (the taker's)
  // and, once one of the trump King/Queen is played, the other.
  std::array<absl::InlinedVector<int, 2>, kNumPlayers> PublicCardPins(
      Player player_id) const;
  // The seat that has announced belote, or kInvalidPlayer if nobody has.
  Player AnnouncedBeloteHolder() const;
  // The counterpart to the belote pin: a seat that played one marriage card
  // without announcing is known not to hold the other. Returns (seat, card),
  // or (kInvalidPlayer, -1).
  std::pair<Player, int> PublicCardBar() const;

  const Player dealer_;
  Hands hands_{};
  // Cards still in the stock, as a membership bitmap: scanning 0..31 in
  // index order is already ascending, so nothing needs sorting.
  std::array<bool, kNumCards> in_deck_{};
  int deck_size_ = 0;
  int turned_card_ = kInvalidAction;

  Phase phase_ = Phase::kDeal;
  DealSchedule deal_schedule_;
  int deal_index_ = 0;
  Phase after_deal_phase_ = Phase::kBid1;

  std::array<Player, kNumPlayers> bid_turn_order_{};
  int bid_pointer_ = 0;
  // Who has passed, per bidding round, in bid order. Needed for perfect
  // recall: `bid_pointer_` resets between rounds, which would leave a bid1
  // information state byte-identical to the bid2 one that follows it (same
  // 5 cards, same turned card, no trump yet). Kept through the play phase,
  // where who passed on which suit is still evidence about the hands held.
  absl::InlinedVector<Player, kNumPlayers> bid1_passes_;
  absl::InlinedVector<Player, kNumPlayers> bid2_passes_;

  Player taker_ = kInvalidPlayer;
  int trump_suit_ = -1;
  int declarer_team_ = -1;
  Player belote_holder_ = kInvalidPlayer;

  Trick trick_;
  Player current_player_play_ = kInvalidPlayer;
  int tricks_played_ = 0;
  absl::InlinedVector<int, kNumCards> played_cards_;
  // Indexed [trick][seat position within that trick], not by player id;
  // ReconstructTricks recovers the player from the leader chain. Only the
  // first tricks_played_ rows are populated, as for trick_winners_.
  std::array<std::array<int, kNumPlayers>, kNumTricks> trick_history_{};
  std::array<Player, kNumTricks> trick_winners_{};
  std::array<int, 2> team_points_ = {0, 0};
  std::array<double, kNumPlayers> returns_{};
};

class BeloteGame : public Game {
 public:
  // Fails if "dealer" is not a seat (0-3).
  explicit BeloteGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumDistinctActions; }
  int MaxChanceOutcomes() const override { return kNumCards; }
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new BeloteState(shared_from_this(), dealer_));
  }
  int NumPlayers() const override { return kNumPlayers; }
  // Loose bounds that also cover a capot (252 instead of 162) and the
  // belote/rebelote bonus.
  double MinUtility() const override {
    return -static_cast<double>(kMaxScoreCapot + kBeloteRebeloteBonus);
  }
  double MaxUtility() const override {
    return static_cast<double>(kMaxScoreCapot + kBeloteRebeloteBonus);
  }
  absl::optional<double> UtilitySum() const override { return 0; }
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override {
    // Dealing the whole deck (32 draws, when someone takes), up to 8 bids,
    // then 32 card plays.
    return kNumCards + 2 * kNumPlayers + kNumCards;
  }

 private:
  const Player dealer_;
};

}  // namespace belote
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_BELOTE_H_

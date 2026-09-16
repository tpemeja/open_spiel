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

// C++ implementation of classic (non-contract) French Belote for 4 players
// in 2 fixed partnerships (players 0 & 2 vs players 1 & 3). Trump is chosen
// via the "prise" procedure: 5 cards are dealt to each player, the next
// stock card is turned face up, and players in turn may take it (round 1)
// or, if everyone passes, choose one of the three other suits (round 2). If
// everyone passes twice, the game ends as a draw (all returns 0): real belote
// would throw the cards in and redeal, but a state here is a single deal,
// and a thrown-in deal scores nothing -- the same choice as bridge's passed
// out hand and euchre without "stick the dealer". Card play follows standard
// suit- and trump-following obligations, and scoring uses the standard
// 162-point deck (152 card points + 10 for the last trick), with an
// all-or-nothing rule: the declaring team keeps its trick points only if it
// scores strictly more than the defenders; otherwise the defending team
// collects all trick points. If one team wins all 8 tricks ("capot"), the
// last-trick bonus is 100 instead of 10, so the deck is worth 252 points
// instead of 162, and that full total goes to whichever team scores higher
// (the capot-winning team on success, or the defenders' 252 on a failed
// contract).
//
// The "belote/rebelote" bonus (20 extra points awarded to whichever team has
// a single player holding both the King and Queen of the trump suit) is
// always applied. Per official rules, this bonus actually requires the
// holder to announce "belote" then "rebelote" when playing the first and
// second of those two cards respectively, and is forfeited if either
// announcement is omitted; this implementation simplifies that away: there
// is no announcement action, and the holder is instead announced
// automatically, to every player, the moment they play the first of the two
// cards (the "belote" field of the observations). Conversely, a player who
// plays one of the two without an announcement is publicly known not to
// hold the other. The 20 points themselves, and the way they're used, do
// follow official rules: they count toward the declaring team's contract
// threshold (its own or the defenders') and are always credited to the
// holder's team, win or lose.

#include <array>
#include <functional>
#include <memory>
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
// Longest possible deal_schedule_: the initial deal (3+2 cards to each of 4
// players, plus the turned card).
inline constexpr int kMaxDealScheduleSize = kNumPlayers * 5 + 1;
inline constexpr int kNumTricks = kNumCards / kNumPlayers;

// Card actions are 0..31 (card = suit * kNumRanks + rank).
inline constexpr int kPassAction = kNumCards;               // 32
inline constexpr int kTakeAction = kNumCards + 1;            // 33
inline constexpr int kChooseSuitActionBase = kNumCards + 2;  // 34..37
inline constexpr int kNumDistinctActions = kNumCards + 2 + kNumSuits;  // 38

constexpr char kSuitChar[] = "CDHS";
// Ranks, low to high face value: 7 8 9 10 J Q K A.
constexpr const char* kRankNames[] = {"7", "8", "9", "10",
                                      "J", "Q", "K", "A"};

enum class Phase { kDeal, kBid1, kBid2, kPlay, kGameOver };

inline int CardSuit(int card) { return card / kNumRanks; }
inline int CardRank(int card) { return card % kNumRanks; }
// The rank name of `card`, e.g. "10", without its suit.
inline std::string CardRankName(int card) { return kRankNames[CardRank(card)]; }
std::string CardString(int card);
int CardPoints(int card, int trump_suit);
int CardStrength(int card, int trump_suit);
// Whether `card` beats `other` within the same trick. Depends only on the
// two cards, the led suit and the trump suit -- not on anything else about
// the deal -- so it is a free function rather than a member, letting a
// strategy weigh a hypothetical trick with no state to hang the call on.
bool Beats(int card, int other, int led_suit, int trump_suit);
inline int TeamOf(Player player) { return player % 2; }
inline Player PartnerOf(Player player) { return (player + 2) % kNumPlayers; }

// A trick in progress or completed, as (player, card) pairs in play order.
// Always at most kNumPlayers entries, so this never allocates on the heap.
using Trick = absl::InlinedVector<std::pair<Player, int>, kNumPlayers>;
// A run of tricks (e.g. every trick played so far, current one included).
// Always at most kNumTricks + 1 entries, so this never allocates either.
using TrickList = absl::InlinedVector<Trick, kNumTricks + 1>;
// One hand of cards per player, indexed by absolute player id. Always at
// most kNumRanks cards, so this never allocates on the heap.
using Hands = std::array<absl::InlinedVector<int, kNumRanks>, kNumPlayers>;
// A deal schedule (which player, or kInvalidPlayer for "turn face up", gets
// each successive card). Always at most kMaxDealScheduleSize entries.
using DealSchedule = absl::InlinedVector<Player, kMaxDealScheduleSize>;

// Void suits and trump-strength upper bounds inferred per player from public
// play, used to constrain opponent hand resampling. `max_trump_strength[p]`
// of -1 means no known bound (any trump strength allowed).
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

  // ---- public read-only view of the state --------------------------------
  //
  // What an agent needs in order to decide anything: the trump suit, whose
  // contract it is, what is on the table and what has already been played.
  // Exposed here rather than left for callers to reach into the private
  // members. The bindings are in games_belote.cc.
  //
  // These return by value, as EuchreState's accessors do. The containers are
  // bounded (at most 32 cards, 8 tricks), so a copy is cheap, and it keeps a
  // caller from holding a reference into a state it may outlive.

  Phase CurrentPhase() const { return phase_; }
  // The phase as a stable name -- "deal", "bid1", "bid2", "play", "done" --
  // for callers that want to test it without depending on the enum's
  // numbering. This is what the bindings expose as current_phase().
  std::string PhaseString() const;
  Player Dealer() const { return dealer_; }
  // The card turned face up for round 1, or absl::nullopt before the deal.
  // Stays set once the auction resolves: every player saw it, and it ends up
  // in the taker's hand.
  absl::optional<int> Upcard() const;
  // The seat holding the contract, or -1 if the auction is unresolved.
  // Belote calls this seat the taker; euchre calls it declarer.
  //
  // Reports -1 rather than the kInvalidPlayer (-3) the member holds, so that
  // an absent seat is spelled the way every other accessor here spells an
  // absent seat, suit or team. A caller testing `>= 0` sees no difference;
  // one testing `== -1` would.
  Player Taker() const { return taker_ >= 0 ? taker_ : -1; }
  int DeclarerTeam() const { return declarer_team_; }
  int TrumpSuit() const { return trump_suit_; }
  // Which auction round is live, or resolved the contract: 1 or 2, and
  // absl::nullopt before the auction opens (during the initial deal), and
  // after four passes in round 2 end the game. CurrentPhase()
  // alone cannot answer this: it reads kBid2 the instant round 2 opens,
  // before anyone in it has acted, and says nothing once the auction is
  // over -- but round 1 having four passes is durable.
  absl::optional<int> BiddingRound() const;
  // Seats that passed in `round_number` (1 or 2), in turn order. Only passes
  // are recorded: a take or a suit call ends the auction, so that seat is
  // never "passed" -- it is Taker().
  std::vector<Player> BidPasses(int round_number) const;
  // (player, card) pairs played so far in the trick in progress.
  std::vector<std::pair<Player, int>> CurrentTrick() const;
  // The winning seat of each completed trick, in order.
  std::vector<Player> TrickWinners() const;
  std::vector<int> PlayedCards() const;
  // Trick points per team. Includes the last-trick bonus, folded in the
  // instant the eighth trick completes. Excludes the belote bonus, which is
  // applied at scoring.
  std::vector<int> TeamPoints() const;
  std::vector<std::vector<int>> PlayerHands() const;
  // The seat holding trump King AND Queen, or kInvalidPlayer. Set in
  // EnterPlayPhase from the real hands, before a card is played: bookkeeping
  // computed when the fact becomes true, not when an opponent could learn
  // it. Under the rules the pair is announced by playing the first of the
  // two cards, so anything shown to a player who is not the holder must gate
  // on BeloteAnnounced().
  // Reports -1 when there is no holder, matching Taker().
  Player BeloteHolder() const {
    return belote_holder_ >= 0 ? belote_holder_ : -1;
  }
  // How many of the two marriage cards the holder has actually played (0-2).
  // The rules-facing counterpart to BeloteHolder(): 0 means the holding is
  // not yet public knowledge, and anything above 0 means it has been
  // announced to every player.
  int BeloteAnnounced() const;
  // Every trick as (player, card) pairs in play order, the trick in progress
  // included.
  std::vector<std::vector<std::pair<Player, int>>> Tricks() const;
  // The King and Queen of the trump suit, or absl::nullopt before there is a
  // trump suit -- the card ids computed from a trump of -1 are negative
  // nonsense rather than an error.
  absl::optional<std::pair<int, int>> TrumpMarriage() const;
  // What the public history proves about the other hands: the suits each
  // seat is known void in, and an upper bound on the trump strength each is
  // known to hold. This is what ResampleFromInfostate uses to keep a sampled
  // world consistent; exposed so a caller sampling or featurising can
  // respect the same constraints instead of re-deriving them.
  VoidAndTrumpBounds PublicInference() const;

 protected:
  void DoApplyAction(Action action) override;

 private:
  std::vector<Action> LegalCardPlays(Player player) const;
  bool IsBetter(int card, int other, int led_suit) const;
  // The (K, Q) card ids of the current trump suit. Only meaningful once a
  // trump suit exists; TrumpMarriage() is the guarded public form.
  std::pair<int, int> TrumpKingAndQueen() const;
  Player TrickWinner(const Trick& trick) const;
  void EnterPlayPhase();
  // Returns the player publicly known -- via `hands`, or having already
  // played the card(s) in `tricks` -- to hold (or have held) both the King
  // and Queen of trump, or kInvalidPlayer. The no-argument overload checks
  // the current hands and play history.
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
  // Backs both InformationStateString and ObservationString. The pieces that
  // are history rather than present state are gated on `perfect_recall` --
  // the same split WriteObservation makes for the tensors.
  std::string ObservationStringImpl(Player player,
                                    bool perfect_recall) const;
  // Rebuilds the completed tricks (not including the current partial trick),
  // as (player, card) pairs in play order. Trick 0 is led by the player
  // after the dealer; trick i>0 is led by the winner of trick i-1.
  TrickList ReconstructCompletedTricks() const;
  // As above, plus the current partial trick (if any) appended at the end.
  TrickList ReconstructTricks() const;
  // Infers void suits and trump-strength upper bounds per player from
  // `tricks`, for constraining opponent hand resampling.
  VoidAndTrumpBounds InferVoidAndTrumpBounds(const TrickList& tricks) const;
  // Cards whose holder is public knowledge beyond `player_id`'s own hand:
  // the turned card (pinned to the taker) and, once exactly one of the
  // trump King/Queen has been publicly played, the other (pinned to the
  // belote holder). At most 2 cards can ever be pinned to one player.
  std::array<absl::InlinedVector<int, 2>, kNumPlayers> PublicCardPins(
      Player player_id) const;
  // The seat that has announced belote, or kInvalidPlayer if nobody has.
  Player AnnouncedBeloteHolder() const;
  // The counterpart to the belote pin: once exactly one of the trump King and
  // Queen has been played, by a seat that did not announce belote, that seat
  // is publicly known not to hold the other. Returns (seat, card), or
  // (kInvalidPlayer, -1) when nothing is known.
  std::pair<Player, int> PublicCardBar() const;

  const Player dealer_;
  Hands hands_{};
  // Cards still in the stock, tracked as a membership bitmap (rather than a
  // vector requiring O(n) erase-by-value and repeated sorting) since cards
  // 0..31 are already in ascending order when scanned in index order.
  std::array<bool, kNumCards> in_deck_{};
  int deck_size_ = 0;
  int turned_card_ = kInvalidAction;

  Phase phase_ = Phase::kDeal;
  DealSchedule deal_schedule_;
  int deal_index_ = 0;
  Phase after_deal_phase_ = Phase::kBid1;

  std::array<Player, kNumPlayers> bid_turn_order_{};
  int bid_pointer_ = 0;
  // Who has passed, per bidding round. `bid_pointer_` alone can't answer
  // this: it resets to 0 between rounds, so without these
  // the auction is unrecoverable from the state -- and a bid1 information
  // state was byte-identical to the bid2 one that follows it (same 5 cards,
  // same turned card, no trump yet), which is a perfect-recall violation: a
  // player could not remember their own pass. Recorded in bid order, and
  // kept through the play phase because who passed on which suit stays
  // public, informative evidence about the hands still held.
  absl::InlinedVector<Player, kNumPlayers> bid1_passes_;
  absl::InlinedVector<Player, kNumPlayers> bid2_passes_;

  Player taker_ = kInvalidPlayer;
  int trump_suit_ = -1;
  int declarer_team_ = -1;
  Player belote_holder_ = kInvalidPlayer;

  Trick trick_;
  Player trick_leader_ = kInvalidPlayer;
  Player current_player_play_ = kInvalidPlayer;
  int tricks_played_ = 0;
  // At most kNumCards entries; never allocates on the heap.
  absl::InlinedVector<int, kNumCards> played_cards_;
  // Cards of each completed trick, indexed [trick][seat position within
  // that trick] (not by absolute player id -- see ReconstructTricks, which
  // recovers the actual player via the trick's leader chain). Only the
  // first tricks_played_ rows are populated.
  std::array<std::array<int, kNumPlayers>, kNumTricks> trick_history_{};
  // Winner of each completed trick; only the first tricks_played_ entries
  // are populated.
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
    return std::unique_ptr<State>(
        new BeloteState(shared_from_this(), dealer_));
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

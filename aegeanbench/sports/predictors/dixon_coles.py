"""
Dixon-Coles bivariate Poisson football model.

Reference: Dixon & Coles (1997), "Modelling Association Football Scores and
Inefficiencies in the Football Betting Market." Journal of the Royal
Statistical Society: Series C.

Model:
    Home goals H ~ Poisson(lambda)
    Away goals A ~ Poisson(mu)

    lambda = exp(alpha_home + beta_away + gamma)   # gamma = home advantage
    mu     = exp(alpha_away + beta_home)

    Joint PMF = tau(H, A, lambda, mu) * Poisson(H|lambda) * Poisson(A|mu)

The tau correction inflates draw probabilities for low scorelines (0-0, 1-1)
and corrects systematic underestimation of low-scoring games by the naive
independent-Poisson model.

Training:
    Pure-python iterative MLE via gradient ascent on the log-likelihood.
    Avoids scipy dependency for sprint; converges in ~50 iterations on
    500 matches.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Optional, Tuple

from aegeanbench.sports.gateway import MatchContext
from aegeanbench.sports.models import (
    Match,
    MatchOutcome,
    Prediction,
)
from aegeanbench.sports.predictors.base import Predictor


def _poisson_pmf(k: int, lam: float) -> float:
    """Poisson probability mass function. k! computed iteratively."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    log_pmf = -lam + k * math.log(lam) - sum(math.log(i) for i in range(1, k + 1))
    return math.exp(log_pmf)


def _tau(home_goals: int, away_goals: int, lam: float, mu: float, rho: float) -> float:
    """
    Dixon-Coles low-score correction.

    Only adjusts the four lowest scorelines (0-0, 1-0, 0-1, 1-1).
    rho < 0 inflates draws and depresses single-goal wins (typical real-world).
    """
    if home_goals == 0 and away_goals == 0:
        return 1.0 - lam * mu * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + lam * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + mu * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


class DixonColesPredictor(Predictor):
    """
    Trains team-level attack and defense strengths from historical results.

    Two parameters per team (alpha = attack, beta = defense) plus two
    global parameters (gamma = home advantage, rho = low-score correction).
    """

    runner_id = "dixon_coles"
    display_name = "Dixon-Coles"

    # Goal range to sum over when computing outcome probabilities.
    # International football matches almost never exceed 6-6.
    MAX_GOALS = 8

    # Hard cap on the log-rate parameters to prevent math.exp overflow during
    # training. Real-world alpha/beta rarely exceed |1.5| (~exp(1.5) = 4.5 goals
    # expectation), so 3.0 is a generous safety bound.
    PARAM_CLAMP = 3.0

    def __init__(
        self,
        home_advantage_init: float = 0.30,
        rho_init: float = -0.10,
        learning_rate: float = 0.003,
        max_iterations: int = 100,
        convergence_eps: float = 1e-4,
    ):
        self.alpha: Dict[str, float] = {}   # attack strength (log scale)
        self.beta: Dict[str, float] = {}    # defense strength (log scale)
        self.gamma: float = home_advantage_init
        self.rho: float = rho_init
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_eps = convergence_eps
        self.fitted = False

    # ---------- training ----------

    def fit(self, history: Iterable[Match]) -> None:
        """
        Fit alpha, beta, gamma, rho via gradient ascent on log-likelihood.

        Numerical pragma: we keep this simple and dependency-free. For
        production-grade fits, swap in scipy.optimize.minimize with L-BFGS-B.
        """
        matches: List[Match] = [m for m in history if m.result is not None]
        if not matches:
            self.fitted = True
            return

        # Initialize team params from observed teams
        teams = set()
        for m in matches:
            teams.add(m.home_team.fifa_code)
            teams.add(m.away_team.fifa_code)
        for code in teams:
            self.alpha.setdefault(code, 0.0)
            self.beta.setdefault(code, 0.0)

        prev_ll = None
        for iteration in range(self.max_iterations):
            ll, grads = self._compute_log_likelihood_and_grads(matches)
            self._apply_grads(grads)

            if prev_ll is not None and abs(ll - prev_ll) < self.convergence_eps:
                break
            prev_ll = ll

        self.fitted = True

    def _compute_log_likelihood_and_grads(
        self, matches: List[Match]
    ) -> Tuple[float, Dict[str, Dict[str, float]]]:
        """
        One pass over matches:
          - accumulate log-likelihood
          - accumulate gradients for alpha/beta/gamma/rho

        Returns (log_likelihood, grads) where grads is a nested dict.
        """
        log_lik = 0.0
        grad_alpha: Dict[str, float] = {k: 0.0 for k in self.alpha}
        grad_beta: Dict[str, float] = {k: 0.0 for k in self.beta}
        grad_gamma = 0.0

        for m in matches:
            hc = m.home_team.fifa_code
            ac = m.away_team.fifa_code
            hg = m.result.home_goals
            ag = m.result.away_goals

            log_lam = self.alpha[hc] + self.beta[ac] + self.gamma
            log_mu = self.alpha[ac] + self.beta[hc]
            # Clamp to prevent overflow; numerical guard only, real values stay well within
            log_lam = max(-self.PARAM_CLAMP, min(self.PARAM_CLAMP, log_lam))
            log_mu = max(-self.PARAM_CLAMP, min(self.PARAM_CLAMP, log_mu))
            lam = math.exp(log_lam)
            mu = math.exp(log_mu)

            # log P(H=hg | lam) + log P(A=ag | mu)
            log_lik += -lam + hg * math.log(max(lam, 1e-12)) - sum(
                math.log(i) for i in range(1, hg + 1)
            )
            log_lik += -mu + ag * math.log(max(mu, 1e-12)) - sum(
                math.log(i) for i in range(1, ag + 1)
            )

            # Gradients: d/d(alpha_hc) log P(H=hg|lam) = hg - lam
            #            d/d(beta_ac)  log P(H=hg|lam) = hg - lam
            #            d/d(gamma)    log P(H=hg|lam) = hg - lam
            grad_alpha[hc] += hg - lam
            grad_beta[ac] += hg - lam
            grad_gamma += hg - lam

            grad_alpha[ac] += ag - mu
            grad_beta[hc] += ag - mu

        return log_lik, {
            "alpha": grad_alpha,
            "beta": grad_beta,
            "gamma": grad_gamma,
        }

    def _apply_grads(self, grads: Dict[str, object]) -> None:
        lr = self.learning_rate
        clamp = self.PARAM_CLAMP
        for code, g in grads["alpha"].items():  # type: ignore[union-attr]
            self.alpha[code] = max(-clamp, min(clamp, self.alpha[code] + lr * g))
        for code, g in grads["beta"].items():   # type: ignore[union-attr]
            self.beta[code] = max(-clamp, min(clamp, self.beta[code] + lr * g))
        self.gamma = max(-clamp, min(clamp, self.gamma + lr * grads["gamma"]))  # type: ignore[operator]

        # Identifiability constraint: alphas and betas each average to 0
        mean_alpha = sum(self.alpha.values()) / len(self.alpha)
        for code in self.alpha:
            self.alpha[code] -= mean_alpha
        mean_beta = sum(self.beta.values()) / len(self.beta)
        for code in self.beta:
            self.beta[code] -= mean_beta

    # ---------- inference ----------

    def predict(self, ctx: MatchContext) -> Prediction:
        start = time.perf_counter()
        m = ctx.match
        hc = m.home_team.fifa_code
        ac = m.away_team.fifa_code

        alpha_h = self.alpha.get(hc, 0.0)
        alpha_a = self.alpha.get(ac, 0.0)
        beta_h = self.beta.get(hc, 0.0)
        beta_a = self.beta.get(ac, 0.0)

        lam = math.exp(alpha_h + beta_a + self.gamma)
        mu = math.exp(alpha_a + beta_h)

        p_home = 0.0
        p_draw = 0.0
        p_away = 0.0
        for h in range(self.MAX_GOALS + 1):
            for a in range(self.MAX_GOALS + 1):
                joint = _tau(h, a, lam, mu, self.rho) * _poisson_pmf(h, lam) * _poisson_pmf(a, mu)
                if h > a:
                    p_home += joint
                elif h < a:
                    p_away += joint
                else:
                    p_draw += joint

        total = p_home + p_draw + p_away
        p_home /= total
        p_draw /= total
        p_away /= total

        latency_ms = int((time.perf_counter() - start) * 1000)

        return Prediction(
            match_id=m.match_id,
            runner_id=self.runner_id,
            p_home_win=p_home,
            p_draw=p_draw,
            p_away_win=p_away,
            confidence=max(p_home, p_draw, p_away),
            rationale=(
                f"Dixon-Coles: lambda={lam:.2f} mu={mu:.2f} "
                f"(alpha_h={alpha_h:+.2f} beta_a={beta_a:+.2f} "
                f"alpha_a={alpha_a:+.2f} beta_h={beta_h:+.2f})"
            ),
            latency_ms=latency_ms,
            metadata={
                "lambda": lam,
                "mu": mu,
                "gamma": self.gamma,
                "rho": self.rho,
            },
        )

    # ---------- introspection ----------

    def team_strength_table(self) -> List[Tuple[str, float, float]]:
        """Return (fifa_code, attack, defense) sorted by net strength."""
        rows = [
            (code, self.alpha.get(code, 0.0), self.beta.get(code, 0.0))
            for code in self.alpha
        ]
        rows.sort(key=lambda r: r[1] - r[2], reverse=True)
        return rows

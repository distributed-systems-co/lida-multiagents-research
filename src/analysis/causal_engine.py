#!/usr/bin/env python3
"""
Causal Inference Engine for LIDA Experiments

Implements:
- Structural Causal Models (SCMs) for debate dynamics
- Do-calculus for intervention effects
- AIPW (Augmented Inverse Propensity Weighted) estimators
- Mediation analysis for mechanism identification
- Heterogeneous treatment effect estimation
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable, Set
import logging
import json

logger = logging.getLogger("lida.causal")


class EstimatorType(Enum):
    """Types of causal effect estimators."""
    NAIVE = "naive"  # Simple difference in means
    IPW = "ipw"  # Inverse propensity weighting
    AIPW = "aipw"  # Augmented IPW (doubly robust)
    MATCHING = "matching"  # Nearest neighbor matching
    REGRESSION = "regression"  # Regression adjustment
    DML = "dml"  # Double machine learning


class TreatmentType(Enum):
    """Types of treatments/interventions in debates."""
    ARGUMENT_INJECTION = "argument_injection"
    SPEAKER_ORDER = "speaker_order"
    INFORMATION_REVEAL = "information_reveal"
    COALITION_FORMATION = "coalition_formation"
    EMOTIONAL_TRIGGER = "emotional_trigger"
    AUTHORITY_APPEAL = "authority_appeal"
    FRAMING_SHIFT = "framing_shift"


@dataclass
class CausalVariable:
    """A variable in the causal graph."""
    name: str
    var_type: str = "continuous"  # continuous, binary, categorical
    is_treatment: bool = False
    is_outcome: bool = False
    is_confounder: bool = False
    is_mediator: bool = False
    domain: Optional[Tuple[float, float]] = None
    categories: Optional[List[str]] = None


@dataclass
class CausalEffect:
    """Estimated causal effect with uncertainty."""
    treatment: str
    outcome: str
    estimator: EstimatorType

    # Point estimates
    ate: float = 0.0  # Average Treatment Effect
    att: float = 0.0  # Average Treatment Effect on Treated
    atc: float = 0.0  # Average Treatment Effect on Control

    # Uncertainty
    se: float = 0.0  # Standard error
    ci_lower: float = 0.0  # 95% CI lower
    ci_upper: float = 0.0  # 95% CI upper
    p_value: float = 1.0

    # Diagnostics
    n_treated: int = 0
    n_control: int = 0
    overlap_score: float = 0.0  # Propensity overlap
    balance_score: float = 0.0  # Covariate balance

    # Heterogeneity
    cate_by_group: Dict[str, float] = field(default_factory=dict)

    # Metadata
    computed_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if effect is statistically significant."""
        return self.p_value < alpha

    def effect_size(self) -> str:
        """Interpret effect size."""
        abs_ate = abs(self.ate)
        if abs_ate < 0.1:
            return "negligible"
        elif abs_ate < 0.3:
            return "small"
        elif abs_ate < 0.5:
            return "medium"
        else:
            return "large"


@dataclass
class StructuralEquation:
    """A structural equation in the SCM."""
    outcome: str
    parents: List[str]
    functional_form: str = "linear"  # linear, polynomial, nonparametric
    coefficients: Dict[str, float] = field(default_factory=dict)
    noise_dist: str = "gaussian"
    noise_params: Dict[str, float] = field(default_factory=lambda: {"mean": 0, "std": 1})

    def evaluate(self, parent_values: Dict[str, float], noise: float = 0.0) -> float:
        """Evaluate the structural equation."""
        if self.functional_form == "linear":
            result = sum(
                self.coefficients.get(p, 0) * parent_values.get(p, 0)
                for p in self.parents
            )
            return result + noise
        elif self.functional_form == "polynomial":
            # Second order terms
            result = 0.0
            for p in self.parents:
                val = parent_values.get(p, 0)
                result += self.coefficients.get(p, 0) * val
                result += self.coefficients.get(f"{p}^2", 0) * val * val
            return result + noise
        else:
            # Nonparametric - would need a trained model
            return noise


@dataclass
class StructuralCausalModel:
    """
    Structural Causal Model for debate dynamics.

    Defines:
    - Endogenous variables (beliefs, positions, emotions)
    - Exogenous noise variables
    - Structural equations relating them
    - Intervention semantics
    """
    name: str
    variables: Dict[str, CausalVariable] = field(default_factory=dict)
    equations: Dict[str, StructuralEquation] = field(default_factory=dict)
    adjacency: Dict[str, Set[str]] = field(default_factory=dict)  # parent -> children

    # Learned from data
    fitted: bool = False
    observed_data: Optional[np.ndarray] = None
    variable_names: List[str] = field(default_factory=list)

    def add_variable(self, var: CausalVariable):
        """Add a variable to the model."""
        self.variables[var.name] = var
        if var.name not in self.adjacency:
            self.adjacency[var.name] = set()

    def add_edge(self, parent: str, child: str):
        """Add a causal edge."""
        if parent not in self.adjacency:
            self.adjacency[parent] = set()
        self.adjacency[parent].add(child)

    def add_equation(self, eq: StructuralEquation):
        """Add a structural equation."""
        self.equations[eq.outcome] = eq
        for parent in eq.parents:
            self.add_edge(parent, eq.outcome)

    def intervene(self, interventions: Dict[str, float]) -> "StructuralCausalModel":
        """
        Create mutilated model with do(X=x) interventions.

        Removes incoming edges to intervened variables and
        sets them to constant values.
        """
        mutilated = StructuralCausalModel(
            name=f"{self.name}_do({list(interventions.keys())})",
            variables=self.variables.copy(),
            equations={},
            adjacency={k: v.copy() for k, v in self.adjacency.items()},
        )

        for var_name, eq in self.equations.items():
            if var_name in interventions:
                # Replace equation with constant
                mutilated.equations[var_name] = StructuralEquation(
                    outcome=var_name,
                    parents=[],
                    coefficients={"_constant": interventions[var_name]},
                )
                # Remove incoming edges
                for parent in eq.parents:
                    if parent in mutilated.adjacency:
                        mutilated.adjacency[parent].discard(var_name)
            else:
                mutilated.equations[var_name] = eq

        return mutilated

    def sample(self, n: int = 1000, interventions: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        """Sample from the SCM (or mutilated version)."""
        model = self if interventions is None else self.intervene(interventions)

        # Topological sort
        order = model._topological_sort()

        samples = {var: np.zeros(n) for var in model.variables}

        for var_name in order:
            if var_name in model.equations:
                eq = model.equations[var_name]

                if "_constant" in eq.coefficients:
                    samples[var_name] = np.full(n, eq.coefficients["_constant"])
                else:
                    # Generate noise
                    if eq.noise_dist == "gaussian":
                        noise = np.random.normal(
                            eq.noise_params.get("mean", 0),
                            eq.noise_params.get("std", 1),
                            n
                        )
                    else:
                        noise = np.zeros(n)

                    # Evaluate equation
                    for i in range(n):
                        parent_vals = {p: samples[p][i] for p in eq.parents}
                        samples[var_name][i] = eq.evaluate(parent_vals, noise[i])
            else:
                # Exogenous - sample from prior
                samples[var_name] = np.random.normal(0, 1, n)

        return samples

    def _topological_sort(self) -> List[str]:
        """Topological sort of variables."""
        in_degree = {v: 0 for v in self.variables}

        for parent, children in self.adjacency.items():
            for child in children:
                in_degree[child] = in_degree.get(child, 0) + 1

        queue = [v for v, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            v = queue.pop(0)
            order.append(v)

            for child in self.adjacency.get(v, set()):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return order

    def fit(self, data: Dict[str, np.ndarray]):
        """Fit structural equations from observational data."""
        self.observed_data = np.column_stack([data[v] for v in self.variables])
        self.variable_names = list(self.variables.keys())

        # Fit each equation via regression
        for var_name, eq in self.equations.items():
            if eq.parents:
                y = data[var_name]
                X = np.column_stack([data[p] for p in eq.parents])

                # OLS fit
                X_bias = np.column_stack([np.ones(len(y)), X])
                coeffs = np.linalg.lstsq(X_bias, y, rcond=None)[0]

                eq.coefficients = {
                    "_bias": coeffs[0],
                    **{p: coeffs[i+1] for i, p in enumerate(eq.parents)}
                }

        self.fitted = True


class CausalEngine:
    """
    Main causal inference engine for experiment analysis.

    Supports:
    - Effect estimation with multiple estimators
    - Propensity score methods
    - Mediation analysis
    - Heterogeneous treatment effects
    - Sensitivity analysis
    """

    def __init__(self, scm: Optional[StructuralCausalModel] = None):
        self.scm = scm or self._build_default_scm()
        self._propensity_model = None
        self._outcome_model = None

    def _build_default_scm(self) -> StructuralCausalModel:
        """Build default SCM for debate dynamics."""
        scm = StructuralCausalModel(name="debate_dynamics")

        # Variables
        variables = [
            CausalVariable("prior_belief", is_confounder=True),
            CausalVariable("expertise", is_confounder=True),
            CausalVariable("emotional_state", is_confounder=True),
            CausalVariable("argument_exposure", is_treatment=True),
            CausalVariable("argument_quality", is_mediator=True),
            CausalVariable("perceived_credibility", is_mediator=True),
            CausalVariable("final_position", is_outcome=True),
            CausalVariable("position_change", is_outcome=True),
        ]

        for v in variables:
            scm.add_variable(v)

        # Structural equations
        scm.add_equation(StructuralEquation(
            outcome="argument_quality",
            parents=["argument_exposure", "expertise"],
            coefficients={"argument_exposure": 0.5, "expertise": 0.3}
        ))

        scm.add_equation(StructuralEquation(
            outcome="perceived_credibility",
            parents=["expertise", "emotional_state"],
            coefficients={"expertise": 0.6, "emotional_state": -0.2}
        ))

        scm.add_equation(StructuralEquation(
            outcome="final_position",
            parents=["prior_belief", "argument_quality", "perceived_credibility"],
            coefficients={"prior_belief": 0.4, "argument_quality": 0.35, "perceived_credibility": 0.25}
        ))

        scm.add_equation(StructuralEquation(
            outcome="position_change",
            parents=["prior_belief", "final_position"],
            coefficients={"prior_belief": -1.0, "final_position": 1.0}
        ))

        return scm

    def estimate_effect(
        self,
        data: Dict[str, np.ndarray],
        treatment: str,
        outcome: str,
        covariates: Optional[List[str]] = None,
        estimator: EstimatorType = EstimatorType.AIPW,
    ) -> CausalEffect:
        """
        Estimate causal effect of treatment on outcome.

        Args:
            data: Dictionary mapping variable names to arrays
            treatment: Name of treatment variable
            outcome: Name of outcome variable
            covariates: Confounders to adjust for
            estimator: Estimation method

        Returns:
            CausalEffect with ATE, ATT, confidence intervals
        """
        T = data[treatment]
        Y = data[outcome]

        if covariates:
            X = np.column_stack([data[c] for c in covariates])
        else:
            X = np.ones((len(T), 1))

        n_treated = int(np.sum(T > 0.5))
        n_control = len(T) - n_treated

        if estimator == EstimatorType.NAIVE:
            return self._estimate_naive(T, Y, treatment, outcome, n_treated, n_control)
        elif estimator == EstimatorType.IPW:
            return self._estimate_ipw(T, Y, X, treatment, outcome, n_treated, n_control)
        elif estimator == EstimatorType.AIPW:
            return self._estimate_aipw(T, Y, X, treatment, outcome, n_treated, n_control)
        elif estimator == EstimatorType.MATCHING:
            return self._estimate_matching(T, Y, X, treatment, outcome, n_treated, n_control)
        else:
            return self._estimate_regression(T, Y, X, treatment, outcome, n_treated, n_control)

    def _estimate_naive(
        self, T: np.ndarray, Y: np.ndarray,
        treatment: str, outcome: str,
        n_treated: int, n_control: int
    ) -> CausalEffect:
        """Simple difference in means."""
        treated_mask = T > 0.5

        mean_treated = np.mean(Y[treated_mask])
        mean_control = np.mean(Y[~treated_mask])
        ate = mean_treated - mean_control

        # Standard error
        var_treated = np.var(Y[treated_mask]) / n_treated if n_treated > 0 else 0
        var_control = np.var(Y[~treated_mask]) / n_control if n_control > 0 else 0
        se = np.sqrt(var_treated + var_control)

        return CausalEffect(
            treatment=treatment,
            outcome=outcome,
            estimator=EstimatorType.NAIVE,
            ate=ate,
            att=ate,
            atc=ate,
            se=se,
            ci_lower=ate - 1.96 * se,
            ci_upper=ate + 1.96 * se,
            p_value=2 * (1 - self._norm_cdf(abs(ate) / se)) if se > 0 else 1.0,
            n_treated=n_treated,
            n_control=n_control,
        )

    def _estimate_ipw(
        self, T: np.ndarray, Y: np.ndarray, X: np.ndarray,
        treatment: str, outcome: str,
        n_treated: int, n_control: int
    ) -> CausalEffect:
        """Inverse propensity weighting."""
        # Fit propensity model (logistic regression)
        propensity = self._fit_propensity(T, X)

        # Clip propensity scores for stability
        propensity = np.clip(propensity, 0.01, 0.99)

        # IPW estimator
        weights_treated = T / propensity
        weights_control = (1 - T) / (1 - propensity)

        ate = np.mean(weights_treated * Y) - np.mean(weights_control * Y)

        # Variance estimation via influence function
        influence = weights_treated * Y - weights_control * Y - ate
        se = np.std(influence) / np.sqrt(len(T))

        return CausalEffect(
            treatment=treatment,
            outcome=outcome,
            estimator=EstimatorType.IPW,
            ate=ate,
            att=ate,  # Simplified
            atc=ate,
            se=se,
            ci_lower=ate - 1.96 * se,
            ci_upper=ate + 1.96 * se,
            p_value=2 * (1 - self._norm_cdf(abs(ate) / se)) if se > 0 else 1.0,
            n_treated=n_treated,
            n_control=n_control,
            overlap_score=self._compute_overlap(propensity, T),
        )

    def _estimate_aipw(
        self, T: np.ndarray, Y: np.ndarray, X: np.ndarray,
        treatment: str, outcome: str,
        n_treated: int, n_control: int
    ) -> CausalEffect:
        """Augmented IPW (doubly robust)."""
        # Fit models
        propensity = self._fit_propensity(T, X)
        propensity = np.clip(propensity, 0.01, 0.99)

        # Outcome models
        mu1 = self._fit_outcome(Y[T > 0.5], X[T > 0.5], X)
        mu0 = self._fit_outcome(Y[T <= 0.5], X[T <= 0.5], X)

        # AIPW estimator
        n = len(T)

        # Potential outcome estimates
        term1 = T * (Y - mu1) / propensity + mu1
        term0 = (1 - T) * (Y - mu0) / (1 - propensity) + mu0

        ate = np.mean(term1 - term0)

        # Influence function variance
        influence = term1 - term0 - ate
        se = np.std(influence) / np.sqrt(n)

        return CausalEffect(
            treatment=treatment,
            outcome=outcome,
            estimator=EstimatorType.AIPW,
            ate=ate,
            att=np.mean((term1 - term0)[T > 0.5]),
            atc=np.mean((term1 - term0)[T <= 0.5]),
            se=se,
            ci_lower=ate - 1.96 * se,
            ci_upper=ate + 1.96 * se,
            p_value=2 * (1 - self._norm_cdf(abs(ate) / se)) if se > 0 else 1.0,
            n_treated=n_treated,
            n_control=n_control,
            overlap_score=self._compute_overlap(propensity, T),
            balance_score=self._compute_balance(X, T, propensity),
        )

    def _estimate_matching(
        self, T: np.ndarray, Y: np.ndarray, X: np.ndarray,
        treatment: str, outcome: str,
        n_treated: int, n_control: int
    ) -> CausalEffect:
        """Nearest neighbor matching on covariates."""
        from scipy.spatial.distance import cdist

        treated_idx = np.where(T > 0.5)[0]
        control_idx = np.where(T <= 0.5)[0]

        if len(treated_idx) == 0 or len(control_idx) == 0:
            return CausalEffect(
                treatment=treatment, outcome=outcome,
                estimator=EstimatorType.MATCHING
            )

        # Find nearest neighbor for each treated unit
        distances = cdist(X[treated_idx], X[control_idx])
        matched_idx = control_idx[np.argmin(distances, axis=1)]

        # ATT via matching
        att = np.mean(Y[treated_idx] - Y[matched_idx])

        # Bootstrap SE
        n_boot = 200
        atts = []
        for _ in range(n_boot):
            boot_idx = np.random.choice(len(treated_idx), len(treated_idx), replace=True)
            boot_treated = treated_idx[boot_idx]
            boot_matched = matched_idx[boot_idx]
            atts.append(np.mean(Y[boot_treated] - Y[boot_matched]))

        se = np.std(atts)

        return CausalEffect(
            treatment=treatment,
            outcome=outcome,
            estimator=EstimatorType.MATCHING,
            ate=att,
            att=att,
            atc=att,
            se=se,
            ci_lower=att - 1.96 * se,
            ci_upper=att + 1.96 * se,
            p_value=2 * (1 - self._norm_cdf(abs(att) / se)) if se > 0 else 1.0,
            n_treated=n_treated,
            n_control=n_control,
        )

    def _estimate_regression(
        self, T: np.ndarray, Y: np.ndarray, X: np.ndarray,
        treatment: str, outcome: str,
        n_treated: int, n_control: int
    ) -> CausalEffect:
        """Regression adjustment."""
        # Y = alpha + beta*T + gamma*X + epsilon
        X_full = np.column_stack([np.ones(len(T)), T, X])

        try:
            coeffs, residuals, _, _ = np.linalg.lstsq(X_full, Y, rcond=None)
            ate = coeffs[1]  # Coefficient on treatment

            # Standard error
            if len(residuals) > 0:
                mse = residuals[0] / (len(Y) - X_full.shape[1])
                var_coeff = mse * np.linalg.inv(X_full.T @ X_full)
                se = np.sqrt(var_coeff[1, 1])
            else:
                se = 0.1  # Fallback

        except np.linalg.LinAlgError:
            ate, se = 0.0, 1.0

        return CausalEffect(
            treatment=treatment,
            outcome=outcome,
            estimator=EstimatorType.REGRESSION,
            ate=ate,
            att=ate,
            atc=ate,
            se=se,
            ci_lower=ate - 1.96 * se,
            ci_upper=ate + 1.96 * se,
            p_value=2 * (1 - self._norm_cdf(abs(ate) / se)) if se > 0 else 1.0,
            n_treated=n_treated,
            n_control=n_control,
        )

    def _fit_propensity(self, T: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Fit propensity score model."""
        # Simple logistic regression
        X_bias = np.column_stack([np.ones(len(T)), X])

        # Initialize
        beta = np.zeros(X_bias.shape[1])

        # Newton-Raphson iterations
        for _ in range(20):
            p = 1 / (1 + np.exp(-X_bias @ beta))
            p = np.clip(p, 1e-6, 1 - 1e-6)

            W = np.diag(p * (1 - p))
            gradient = X_bias.T @ (T - p)
            hessian = -X_bias.T @ W @ X_bias

            try:
                beta -= np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                break

        return 1 / (1 + np.exp(-X_bias @ beta))

    def _fit_outcome(
        self,
        Y_subset: np.ndarray,
        X_subset: np.ndarray,
        X_full: np.ndarray
    ) -> np.ndarray:
        """Fit outcome model and predict for all units."""
        if len(Y_subset) < 2:
            return np.full(len(X_full), np.mean(Y_subset) if len(Y_subset) > 0 else 0)

        X_bias = np.column_stack([np.ones(len(Y_subset)), X_subset])

        try:
            coeffs = np.linalg.lstsq(X_bias, Y_subset, rcond=None)[0]
            X_full_bias = np.column_stack([np.ones(len(X_full)), X_full])
            return X_full_bias @ coeffs
        except np.linalg.LinAlgError:
            return np.full(len(X_full), np.mean(Y_subset))

    def _compute_overlap(self, propensity: np.ndarray, T: np.ndarray) -> float:
        """Compute propensity score overlap."""
        treated_prop = propensity[T > 0.5]
        control_prop = propensity[T <= 0.5]

        if len(treated_prop) == 0 or len(control_prop) == 0:
            return 0.0

        # Overlap as min density region
        overlap_region = (
            max(treated_prop.min(), control_prop.min()),
            min(treated_prop.max(), control_prop.max())
        )

        in_overlap = np.mean(
            (propensity >= overlap_region[0]) & (propensity <= overlap_region[1])
        )

        return float(in_overlap)

    def _compute_balance(
        self,
        X: np.ndarray,
        T: np.ndarray,
        propensity: np.ndarray
    ) -> float:
        """Compute covariate balance (weighted standardized mean difference)."""
        weights = np.where(T > 0.5, 1/propensity, 1/(1-propensity))

        # Weighted means
        treated_mean = np.average(X[T > 0.5], weights=weights[T > 0.5], axis=0)
        control_mean = np.average(X[T <= 0.5], weights=weights[T <= 0.5], axis=0)

        # Pooled std
        pooled_std = np.sqrt(
            (np.var(X[T > 0.5], axis=0) + np.var(X[T <= 0.5], axis=0)) / 2
        )
        pooled_std = np.where(pooled_std < 1e-6, 1.0, pooled_std)

        # Standardized mean difference
        smd = np.abs(treated_mean - control_mean) / pooled_std

        # Average balance (lower is better)
        return 1.0 - float(np.mean(smd))

    def _norm_cdf(self, x: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

    def mediation_analysis(
        self,
        data: Dict[str, np.ndarray],
        treatment: str,
        mediator: str,
        outcome: str,
        covariates: Optional[List[str]] = None,
    ) -> Dict[str, CausalEffect]:
        """
        Mediation analysis decomposing total effect.

        Returns:
            Dictionary with 'total', 'direct', and 'indirect' effects
        """
        # Total effect: T -> Y
        total = self.estimate_effect(data, treatment, outcome, covariates)

        # Direct effect: T -> Y controlling for M
        direct_covariates = (covariates or []) + [mediator]
        direct = self.estimate_effect(data, treatment, outcome, direct_covariates)

        # Indirect effect = Total - Direct
        indirect_ate = total.ate - direct.ate
        indirect_se = np.sqrt(total.se**2 + direct.se**2)  # Conservative

        indirect = CausalEffect(
            treatment=treatment,
            outcome=f"{outcome} via {mediator}",
            estimator=total.estimator,
            ate=indirect_ate,
            se=indirect_se,
            ci_lower=indirect_ate - 1.96 * indirect_se,
            ci_upper=indirect_ate + 1.96 * indirect_se,
            p_value=2 * (1 - self._norm_cdf(abs(indirect_ate) / indirect_se)) if indirect_se > 0 else 1.0,
        )

        return {
            "total": total,
            "direct": direct,
            "indirect": indirect,
            "mediation_proportion": indirect_ate / total.ate if abs(total.ate) > 1e-6 else 0.0,
        }

    def heterogeneous_effects(
        self,
        data: Dict[str, np.ndarray],
        treatment: str,
        outcome: str,
        effect_modifiers: List[str],
        covariates: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, CausalEffect]]:
        """
        Estimate heterogeneous treatment effects by subgroups.

        Args:
            effect_modifiers: Variables to stratify by

        Returns:
            Dictionary mapping modifier -> {group -> CausalEffect}
        """
        results = {}

        for modifier in effect_modifiers:
            modifier_data = data[modifier]

            # Split into groups (binary or median split)
            if np.unique(modifier_data).shape[0] <= 2:
                groups = {"low": modifier_data <= 0.5, "high": modifier_data > 0.5}
            else:
                median = np.median(modifier_data)
                groups = {"below_median": modifier_data <= median, "above_median": modifier_data > median}

            results[modifier] = {}

            for group_name, mask in groups.items():
                subset = {k: v[mask] for k, v in data.items()}
                effect = self.estimate_effect(subset, treatment, outcome, covariates)
                results[modifier][group_name] = effect

        return results

    def sensitivity_analysis(
        self,
        effect: CausalEffect,
        data: Dict[str, np.ndarray],
        treatment: str,
        outcome: str,
        gamma_range: Tuple[float, float] = (1.0, 2.0),
        n_points: int = 10,
    ) -> List[Dict[str, float]]:
        """
        Sensitivity analysis for unmeasured confounding.

        Uses Rosenbaum bounds to assess robustness.

        Args:
            gamma_range: Range of sensitivity parameter (1 = no confounding)

        Returns:
            List of (gamma, lower_bound, upper_bound) tuples
        """
        gammas = np.linspace(gamma_range[0], gamma_range[1], n_points)
        results = []

        for gamma in gammas:
            # Rosenbaum bounds adjustment
            # This is a simplified version
            adjustment = np.log(gamma)

            lower = effect.ate - adjustment * effect.se * 2
            upper = effect.ate + adjustment * effect.se * 2

            results.append({
                "gamma": gamma,
                "lower_bound": lower,
                "upper_bound": upper,
                "includes_zero": lower <= 0 <= upper,
            })

        return results


def create_debate_scm() -> StructuralCausalModel:
    """Create a pre-configured SCM for debate analysis."""
    engine = CausalEngine()
    return engine.scm


if __name__ == "__main__":
    # Demo
    np.random.seed(42)
    n = 1000

    # Simulate data
    prior = np.random.normal(0.5, 0.2, n)
    expertise = np.random.uniform(0, 1, n)
    treatment = (np.random.random(n) < (0.3 + 0.4 * expertise)).astype(float)
    quality = 0.5 * treatment + 0.3 * expertise + np.random.normal(0, 0.1, n)
    outcome = 0.4 * prior + 0.35 * quality + 0.1 * treatment + np.random.normal(0, 0.15, n)

    data = {
        "prior_belief": prior,
        "expertise": expertise,
        "argument_exposure": treatment,
        "argument_quality": quality,
        "final_position": outcome,
    }

    engine = CausalEngine()

    # Estimate effects
    print("=== Causal Effect Estimation ===\n")

    for est_type in [EstimatorType.NAIVE, EstimatorType.IPW, EstimatorType.AIPW]:
        effect = engine.estimate_effect(
            data,
            treatment="argument_exposure",
            outcome="final_position",
            covariates=["prior_belief", "expertise"],
            estimator=est_type,
        )
        print(f"{est_type.value.upper()}:")
        print(f"  ATE = {effect.ate:.4f} (SE: {effect.se:.4f})")
        print(f"  95% CI: [{effect.ci_lower:.4f}, {effect.ci_upper:.4f}]")
        print(f"  p-value: {effect.p_value:.4f}")
        print()

    # Mediation analysis
    print("=== Mediation Analysis ===\n")
    mediation = engine.mediation_analysis(
        data,
        treatment="argument_exposure",
        mediator="argument_quality",
        outcome="final_position",
        covariates=["prior_belief", "expertise"],
    )

    print(f"Total effect: {mediation['total'].ate:.4f}")
    print(f"Direct effect: {mediation['direct'].ate:.4f}")
    print(f"Indirect effect: {mediation['indirect'].ate:.4f}")
    print(f"Mediation proportion: {mediation['mediation_proportion']:.1%}")

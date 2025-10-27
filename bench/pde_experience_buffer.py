"""
Experience Buffer for PDE Discovery

Maintains memory of (equation, score, visual_analysis, reasoning) tuples
to enable in-context learning for the equation generator agent.
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime
import heapq


@dataclass
class PDEExperience:
    """Single experience entry for PDE discovery"""
    equation: str
    score: float
    metrics: Dict[str, float]  # mse, r2, nmse, etc. (may also include 'visual_score', 'combined_score')
    visual_analysis: str  # Analysis from Visual Critic
    reasoning: str  # Reasoning from Generator
    suggestions: str  # Improvement suggestions from Critic
    parameters: Dict[str, float]  # Fitted parameters
    timestamp: str
    iteration: int
    # Optional, newly added fields (backward compatible)
    visual_score: Optional[float] = None
    combined_score: Optional[float] = None
    visualization_path: Optional[str] = None
    spatial_assessment: Optional[str] = ""
    temporal_assessment: Optional[str] = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'PDEExperience':
        """Create from dictionary with backward-compatible defaults"""
        return cls(
            equation=data.get('equation', ''),
            score=data.get('score', 0.0),
            metrics=data.get('metrics', {}),
            visual_analysis=data.get('visual_analysis', ''),
            reasoning=data.get('reasoning', ''),
            suggestions=data.get('suggestions', ''),
            parameters=data.get('parameters', {}),
            timestamp=data.get('timestamp', ''),
            iteration=int(data.get('iteration', 0)),
            visual_score=data.get('visual_score'),
            combined_score=data.get('combined_score'),
            visualization_path=data.get('visualization_path'),
            spatial_assessment=data.get('spatial_assessment', ''),
            temporal_assessment=data.get('temporal_assessment', ''),
        )

    def to_prompt_context(self, include_visual: bool = False) -> str:
        """
        Format experience as prompt context

        Args:
            include_visual: Whether to include visual analysis

        Returns:
            Formatted string for prompt injection
        """
        # Format parameters separately to avoid nested f-string issues
        params_str = ', '.join(f'{k}={v:.4f}' for k, v in self.parameters.items())

        # Format metrics safely
        mse_str = f"{self.metrics['mse']:.6f}" if 'mse' in self.metrics else 'N/A'
        r2_str = f"{self.metrics['r2']:.4f}" if 'r2' in self.metrics else 'N/A'
        nmse_str = f"{self.metrics['nmse']:.4f}" if 'nmse' in self.metrics else 'N/A'
        vis_score_str = (
            f"{self.metrics['visual_score']:.2f}" if 'visual_score' in self.metrics and self.metrics['visual_score'] is not None else (
                f"{self.visual_score:.2f}" if self.visual_score is not None else 'N/A'
            )
        )
        combined_str = (
            f"{self.metrics['combined_score']:.2f}" if 'combined_score' in self.metrics and self.metrics['combined_score'] is not None else (
                f"{self.combined_score:.2f}" if self.combined_score is not None else 'N/A'
            )
        )

        context = f"""
Previous Attempt #{self.iteration}:
Equation: {self.equation}
Numerical Score: {self.score:.4f} (0-10)
Metrics: MSE={mse_str}, R²={r2_str}, NMSE={nmse_str}, VisualScore={vis_score_str}, Combined={combined_str}
Parameters: {params_str}
Reasoning: {self.reasoning}
"""
        if include_visual and (self.visual_analysis or self.spatial_assessment or self.temporal_assessment):
            context += "Visual Critic Analysis: "
            if self.visual_analysis:
                context += f"{self.visual_analysis}\n"
            else:
                parts = []
                if self.spatial_assessment:
                    parts.append(f"Spatial: {self.spatial_assessment}")
                if self.temporal_assessment:
                    parts.append(f"Temporal: {self.temporal_assessment}")
                context += (" | ".join(parts) + "\n") if parts else "\n"

        if self.suggestions:
            context += f"Suggestions: {self.suggestions}\n"

        return context.strip()


class PDEExperienceBuffer:
    """
    Experience buffer for PDE discovery with retrieval and pruning

    Features:
    - Store equation attempts with scores and analyses
    - Retrieve top-K entries by score
    - Prune to keep only best/diverse examples
    - Save/load from disk
    """

    def __init__(self, max_size: int = 100, diversity_threshold: float = 0.3):
        """
        Initialize experience buffer

        Args:
            max_size: Maximum number of experiences to store
            diversity_threshold: Minimum string similarity for diversity pruning
        """
        self.max_size = max_size
        self.diversity_threshold = diversity_threshold
        self.experiences: List[PDEExperience] = []
        self.iteration_counter = 0

    def add(self, equation: str, score: float, metrics: Dict[str, float],
            visual_analysis: str = "", reasoning: str = "",
            suggestions: str = "", parameters: Dict[str, float] = None,
            visual_score: Optional[float] = None,
            combined_score: Optional[float] = None,
            visualization_path: Optional[str] = None,
            spatial_assessment: str = "", temporal_assessment: str = "") -> None:
        """
        Add new experience to buffer

        Args:
            equation: PDE equation string
            score: Numerical score (higher is better)
            metrics: Dictionary of evaluation metrics
            visual_analysis: Analysis from Visual Critic
            reasoning: Reasoning from Generator
            suggestions: Improvement suggestions
            parameters: Fitted PDE parameters
        """
        self.iteration_counter += 1

        experience = PDEExperience(
            equation=equation,
            score=score,
            metrics=metrics,
            visual_analysis=visual_analysis,
            reasoning=reasoning,
            suggestions=suggestions,
            parameters=parameters or {},
            timestamp=datetime.now().isoformat(),
            iteration=self.iteration_counter,
            visual_score=visual_score,
            combined_score=combined_score,
            visualization_path=visualization_path,
            spatial_assessment=spatial_assessment,
            temporal_assessment=temporal_assessment,
        )

        self.experiences.append(experience)

        # Prune if exceeds max size
        if len(self.experiences) > self.max_size:
            self._prune()

    def get_top_k(self, k: int = 5, include_visual: bool = False) -> List[PDEExperience]:
        """
        Retrieve top-K experiences by score

        Args:
            k: Number of top experiences to retrieve
            include_visual: Whether experiences need visual analysis

        Returns:
            List of top-K experiences
        """
        if include_visual:
            # Filter to only those with visual analysis
            valid_exps = [exp for exp in self.experiences if exp.visual_analysis]
        else:
            valid_exps = self.experiences

        # Sort by score (descending)
        sorted_exps = sorted(valid_exps, key=lambda x: x.score, reverse=True)

        return sorted_exps[:k]

    def get_best(self) -> Optional[PDEExperience]:
        """Get best experience by score"""
        if not self.experiences:
            return None
        return max(self.experiences, key=lambda x: x.score)

    def get_worst(self) -> Optional[PDEExperience]:
        """Get worst experience by score"""
        if not self.experiences:
            return None
        return min(self.experiences, key=lambda x: x.score)

    def get_recent(self, n: int = 5) -> List[PDEExperience]:
        """Get n most recent experiences"""
        return sorted(self.experiences, key=lambda x: x.iteration, reverse=True)[:n]

    def format_for_prompt(self, k: int = 5, include_visual: bool = False) -> str:
        """
        Format top-K experiences as prompt context

        Args:
            k: Number of experiences to include
            include_visual: Whether to include visual analyses

        Returns:
            Formatted string for prompt injection
        """
        top_exps = self.get_top_k(k, include_visual)

        if not top_exps:
            return "No previous experiences available."

        context = "## Previous Best Attempts:\n\n"
        for exp in top_exps:
            context += exp.to_prompt_context(include_visual) + "\n\n"

        return context.strip()

    def _prune(self) -> None:
        """
        Prune buffer to max_size using diversity-aware selection

        Strategy:
        1. Keep top 20% by score
        2. From remainder, select diverse examples
        3. Remove duplicates and very similar equations
        """
        if len(self.experiences) <= self.max_size:
            return

        # Sort by score
        sorted_exps = sorted(self.experiences, key=lambda x: x.score, reverse=True)

        # Keep top performers
        n_elite = max(int(0.2 * self.max_size), 5)
        elite = sorted_exps[:n_elite]

        # Select diverse examples from remainder
        remaining = sorted_exps[n_elite:]
        diverse_selected = self._select_diverse(remaining, self.max_size - n_elite)

        self.experiences = elite + diverse_selected

    def _select_diverse(self, experiences: List[PDEExperience], n: int) -> List[PDEExperience]:
        """
        Select n diverse experiences using greedy diversity selection

        Args:
            experiences: List of experiences to select from
            n: Number to select

        Returns:
            Selected diverse experiences
        """
        if len(experiences) <= n:
            return experiences

        selected = [experiences[0]]  # Start with first (highest score in remaining)
        candidates = experiences[1:]

        while len(selected) < n and candidates:
            # Select candidate most different from selected set
            max_min_dist = -1
            best_candidate = None
            best_idx = -1

            for idx, candidate in enumerate(candidates):
                # Compute minimum distance to selected set
                min_dist = min(
                    self._equation_distance(candidate.equation, sel.equation)
                    for sel in selected
                )

                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_candidate = candidate
                    best_idx = idx

            if best_candidate and max_min_dist >= self.diversity_threshold:
                selected.append(best_candidate)
                candidates.pop(best_idx)
            else:
                # If no sufficiently diverse candidate, break
                break

        return selected

    def _equation_distance(self, eq1: str, eq2: str) -> float:
        """
        Compute normalized edit distance between equations

        Args:
            eq1, eq2: Equation strings

        Returns:
            Distance in [0, 1] (1 = completely different)
        """
        # Simple character-level Levenshtein distance
        # Normalize by max length
        if eq1 == eq2:
            return 0.0

        m, n = len(eq1), len(eq2)
        if m == 0 or n == 0:
            return 1.0

        # DP for edit distance
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if eq1[i-1] == eq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        edit_dist = dp[m][n]
        max_len = max(m, n)

        return edit_dist / max_len

    def get_statistics(self) -> Dict:
        """
        Get statistics about buffer contents

        Returns:
            Dictionary with statistics
        """
        if not self.experiences:
            return {
                'count': 0,
                'best_score': None,
                'worst_score': None,
                'mean_score': None,
                'unique_equations': 0
            }

        scores = [exp.score for exp in self.experiences]
        equations = [exp.equation for exp in self.experiences]

        return {
            'count': len(self.experiences),
            'best_score': max(scores),
            'worst_score': min(scores),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'unique_equations': len(set(equations)),
            'iterations': self.iteration_counter
        }

    def save(self, filepath: str) -> None:
        """
        Save buffer to JSON file

        Args:
            filepath: Path to save file
        """
        data = {
            'max_size': self.max_size,
            'diversity_threshold': self.diversity_threshold,
            'iteration_counter': self.iteration_counter,
            'experiences': [exp.to_dict() for exp in self.experiences]
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'PDEExperienceBuffer':
        """
        Load buffer from JSON file

        Args:
            filepath: Path to load from

        Returns:
            Loaded PDEExperienceBuffer
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        buffer = cls(
            max_size=data['max_size'],
            diversity_threshold=data['diversity_threshold']
        )
        buffer.iteration_counter = data['iteration_counter']
        buffer.experiences = [
            PDEExperience.from_dict(exp_data)
            for exp_data in data['experiences']
        ]

        return buffer

    def clear(self) -> None:
        """Clear all experiences"""
        self.experiences = []
        self.iteration_counter = 0

    def __len__(self) -> int:
        """Return number of experiences"""
        return len(self.experiences)

    def __repr__(self) -> str:
        """String representation"""
        stats = self.get_statistics()
        return (f"PDEExperienceBuffer(size={len(self)}/{self.max_size}, "
                f"best_score={stats['best_score']}, "
                f"iterations={stats['iterations']})")

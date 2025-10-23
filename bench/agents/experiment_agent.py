from typing import List, Dict, Any
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from ..agents.base import BaseAgent, AgentMessage
from ..dataclasses import SEDTask, SearchResult, Equation
from ..searchers.base import BaseSearcher

class ExperimentAgent(BaseAgent):
    """实验分析智能体 - 专注于从数据角度分析问题"""
    
    def __init__(self, agent_id: str, name: str, llm_sampler, config):
        super().__init__(agent_id, name)
        self.llm_sampler = llm_sampler
        self.config = config
        self.data_analysis_tools = self._initialize_analysis_tools()
    
    def _initialize_analysis_tools(self) -> Dict[str, Any]:
        """初始化数据分析工具"""
        return {
            "correlation_methods": ["pearson", "spearman", "kendall"],
            "regression_methods": ["linear", "polynomial", "exponential", "logarithmic"],
            "pattern_detection": ["trend", "seasonality", "outliers", "clusters"],
            "statistical_tests": ["normality", "homoscedasticity", "independence"]
        }
    
    def process_task(self, task: SEDTask) -> List[SearchResult]:
        """处理任务 - 实验分析"""
        self.state.current_task = task
        
        # 1. 数据探索性分析
        data_analysis = self._explore_data(task)
        
        # 2. 统计模式识别
        patterns = self._identify_patterns(task, data_analysis)
        
        # 3. 基于数据的方程拟合
        fitted_equations = self._fit_equations_from_data(task, patterns)
        
        # 4. 转换为SearchResult格式
        results = []
        for eq_data in fitted_equations:
            equation = Equation(
                symbols=task.symbols,
                symbol_descs=task.symbol_descs,
                symbol_properties=task.symbol_properties,
                expression=eq_data["expression"],
                program_format=eq_data.get("program_format"),
                lambda_format=eq_data.get("lambda_format")
            )
            
            result = SearchResult(
                equation=equation,
                aux={
                    "agent_type": "experiment",
                    "confidence": eq_data.get("confidence", 0.5),
                    "r_squared": eq_data.get("r_squared", 0.0),
                    "pattern_type": eq_data.get("pattern_type", ""),
                    "data_analysis": data_analysis,
                    "statistical_significance": eq_data.get("statistical_significance", 0.0)
                }
            )
            results.append(result)
        
        return results
    
    def _explore_data(self, task: SEDTask) -> Dict[str, Any]:
        """探索性数据分析"""
        data = task.samples
        X = data[:, 1:]  # 输入变量
        y = data[:, 0]   # 输出变量
        
        analysis = {
            "data_shape": data.shape,
            "input_dim": X.shape[1],
            "sample_size": len(y),
            "output_stats": {
                "mean": np.mean(y),
                "std": np.std(y),
                "min": np.min(y),
                "max": np.max(y),
                "range": np.max(y) - np.min(y)
            },
            "correlations": {},
            "data_quality": {}
        }
        
        # 计算相关性
        for i in range(X.shape[1]):
            corr_coef, p_value = stats.pearsonr(X[:, i], y)
            analysis["correlations"][f"x{i}"] = {
                "correlation": corr_coef,
                "p_value": p_value,
                "strength": self._interpret_correlation(abs(corr_coef))
            }
        
        # 数据质量检查
        analysis["data_quality"] = {
            "missing_values": np.isnan(data).sum(),
            "outliers": self._detect_outliers(y),
            "normality": self._test_normality(y)
        }
        
        return analysis
    
    def _interpret_correlation(self, corr: float) -> str:
        """解释相关性强度"""
        if corr >= 0.8:
            return "very_strong"
        elif corr >= 0.6:
            return "strong"
        elif corr >= 0.4:
            return "moderate"
        elif corr >= 0.2:
            return "weak"
        else:
            return "very_weak"
    
    def _detect_outliers(self, y: np.ndarray) -> Dict[str, Any]:
        """检测异常值"""
        Q1 = np.percentile(y, 25)
        Q3 = np.percentile(y, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = np.where((y < lower_bound) | (y > upper_bound))[0]
        
        return {
            "count": len(outliers),
            "percentage": len(outliers) / len(y) * 100,
            "indices": outliers.tolist()
        }
    
    def _test_normality(self, y: np.ndarray) -> Dict[str, Any]:
        """测试正态性"""
        try:
            stat, p_value = stats.normaltest(y)
            return {
                "statistic": stat,
                "p_value": p_value,
                "is_normal": p_value > 0.05
            }
        except:
            return {"is_normal": False, "error": "test_failed"}
    
    def _identify_patterns(self, task: SEDTask, data_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别数据模式"""
        patterns = []
        X = task.samples[:, 1:]
        y = task.samples[:, 0]
        
        # 线性模式
        if self._test_linear_pattern(X, y):
            patterns.append({
                "type": "linear",
                "description": "线性关系",
                "confidence": self._calculate_pattern_confidence(X, y, "linear")
            })
        
        # 多项式模式
        poly_degree = self._test_polynomial_pattern(X, y)
        if poly_degree > 1:
            patterns.append({
                "type": f"polynomial_degree_{poly_degree}",
                "description": f"{poly_degree}次多项式关系",
                "confidence": self._calculate_pattern_confidence(X, y, f"polynomial_{poly_degree}")
            })
        
        # 指数模式
        if self._test_exponential_pattern(X, y):
            patterns.append({
                "type": "exponential",
                "description": "指数关系",
                "confidence": self._calculate_pattern_confidence(X, y, "exponential")
            })
        
        # 周期性模式
        if self._test_periodic_pattern(X, y):
            patterns.append({
                "type": "periodic",
                "description": "周期性关系",
                "confidence": self._calculate_pattern_confidence(X, y, "periodic")
            })
        
        return patterns
    
    def _test_linear_pattern(self, X: np.ndarray, y: np.ndarray) -> bool:
        """测试线性模式"""
        try:
            model = LinearRegression()
            model.fit(X, y)
            r_squared = model.score(X, y)
            return r_squared > 0.7
        except:
            return False
    
    def _test_polynomial_pattern(self, X: np.ndarray, y: np.ndarray) -> int:
        """测试多项式模式"""
        best_degree = 1
        best_r_squared = 0
        
        for degree in range(1, 4):  # 测试1-3次多项式
            try:
                poly_features = PolynomialFeatures(degree=degree)
                X_poly = poly_features.fit_transform(X)
                model = LinearRegression()
                model.fit(X_poly, y)
                r_squared = model.score(X_poly, y)
                
                if r_squared > best_r_squared:
                    best_r_squared = r_squared
                    best_degree = degree
            except:
                continue
        
        return best_degree if best_r_squared > 0.8 else 1
    
    def _test_exponential_pattern(self, X: np.ndarray, y: np.ndarray) -> bool:
        """测试指数模式"""
        try:
            # 检查y是否为正数
            if np.any(y <= 0):
                return False
            
            # 对数变换后测试线性关系
            log_y = np.log(y)
            model = LinearRegression()
            model.fit(X, log_y)
            r_squared = model.score(X, log_y)
            return r_squared > 0.7
        except:
            return False
    
    def _test_periodic_pattern(self, X: np.ndarray, y: np.ndarray) -> bool:
        """测试周期性模式"""
        try:
            # 简单的周期性检测
            if X.shape[1] == 1:  # 单变量情况
                x = X.flatten()
                # 使用FFT检测周期性
                fft = np.fft.fft(y)
                freqs = np.fft.fftfreq(len(y))
                dominant_freq = freqs[np.argmax(np.abs(fft[1:])) + 1]
                return abs(dominant_freq) > 0.1  # 阈值可调
            return False
        except:
            return False
    
    def _calculate_pattern_confidence(self, X: np.ndarray, y: np.ndarray, pattern_type: str) -> float:
        """计算模式置信度"""
        try:
            if pattern_type == "linear":
                model = LinearRegression()
                model.fit(X, y)
                return model.score(X, y)
            elif pattern_type.startswith("polynomial"):
                degree = int(pattern_type.split("_")[1])
                poly_features = PolynomialFeatures(degree=degree)
                X_poly = poly_features.fit_transform(X)
                model = LinearRegression()
                model.fit(X_poly, y)
                return model.score(X_poly, y)
            elif pattern_type == "exponential":
                log_y = np.log(y)
                model = LinearRegression()
                model.fit(X, log_y)
                return model.score(X, log_y)
            else:
                return 0.5
        except:
            return 0.0
    
    def _fit_equations_from_data(self, task: SEDTask, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于数据拟合方程"""
        equations = []
        X = task.samples[:, 1:]
        y = task.samples[:, 0]
        
        # 为每个识别的模式拟合方程
        for i, pattern in enumerate(patterns):
            try:
                eq_data = self._fit_specific_pattern(X, y, pattern, task)
                if eq_data:
                    eq_data["pattern_type"] = pattern["type"]
                    eq_data["confidence"] = pattern["confidence"]
                    equations.append(eq_data)
            except Exception as e:
                print(f"Error fitting pattern {pattern['type']}: {e}")
                continue
        
        # 使用LLM进行数据驱动的方程发现
        llm_equations = self._llm_data_driven_discovery(task, patterns)
        equations.extend(llm_equations)
        
        return equations
    
    def _fit_specific_pattern(self, X: np.ndarray, y: np.ndarray, pattern: Dict[str, Any], task: SEDTask) -> Optional[Dict[str, Any]]:
        """拟合特定模式的方程"""
        pattern_type = pattern["type"]
        
        if pattern_type == "linear":
            model = LinearRegression()
            model.fit(X, y)
            r_squared = model.score(X, y)
            
            # 构建方程字符串
            coefs = model.coef_
            intercept = model.intercept_
            
            equation_parts = [f"{intercept:.4f}"]
            for i, coef in enumerate(coefs):
                equation_parts.append(f"{coef:.4f}*{task.symbols[i+1]}")
            
            equation_str = f"{task.symbols[0]} = " + " + ".join(equation_parts)
            
            return {
                "expression": equation_str,
                "r_squared": r_squared,
                "statistical_significance": r_squared,
                "program_format": equation_str
            }
        
        elif pattern_type.startswith("polynomial"):
            degree = int(pattern_type.split("_")[1])
            poly_features = PolynomialFeatures(degree=degree)
            X_poly = poly_features.fit_transform(X)
            model = LinearRegression()
            model.fit(X_poly, y)
            r_squared = model.score(X_poly, y)
            
            # 简化的多项式方程表示
            equation_str = f"{task.symbols[0]} = polynomial_{degree}({', '.join(task.symbols[1:])})"
            
            return {
                "expression": equation_str,
                "r_squared": r_squared,
                "statistical_significance": r_squared,
                "program_format": equation_str
            }
        
        elif pattern_type == "exponential":
            log_y = np.log(y)
            model = LinearRegression()
            model.fit(X, log_y)
            r_squared = model.score(X, log_y)
            
            # 构建指数方程
            coefs = model.coef_
            intercept = model.intercept_
            
            equation_parts = []
            for i, coef in enumerate(coefs):
                equation_parts.append(f"{coef:.4f}*{task.symbols[i+1]}")
            
            exponent = " + ".join(equation_parts) if equation_parts else "0"
            equation_str = f"{task.symbols[0]} = {np.exp(intercept):.4f} * exp({exponent})"
            
            return {
                "expression": equation_str,
                "r_squared": r_squared,
                "statistical_significance": r_squared,
                "program_format": equation_str
            }
        
        return None
    
    def _llm_data_driven_discovery(self, task: SEDTask, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用LLM进行数据驱动的方程发现"""
        equations = []
        
        # 构建数据驱动的提示
        prompt = self._build_data_driven_prompt(task, patterns)
        
        # 使用LLM生成方程
        for i in range(self.config.samples_per_prompt):
            try:
                response = self.llm_sampler.sample_program(prompt)
                equation_str = response[0] if isinstance(response, tuple) else response
                
                # 解析和验证方程
                parsed_eq = self._parse_equation(equation_str, task)
                if parsed_eq:
                    parsed_eq["pattern_type"] = f"llm_data_driven_{i}"
                    parsed_eq["confidence"] = 0.4  # LLM数据驱动的默认置信度
                    equations.append(parsed_eq)
            except Exception as e:
                print(f"Experiment agent LLM error: {e}")
                continue
        
        return equations
    
    def _build_data_driven_prompt(self, task: SEDTask, patterns: List[Dict[str, Any]]) -> str:
        """构建数据驱动的提示"""
        prompt = f"""
你是一个数据分析专家，需要基于给定的数据模式发现数学方程：

问题描述：
- 符号: {task.symbols}
- 符号描述: {task.symbol_descs}
- 数据形状: {task.samples.shape}

通过统计分析，我发现了以下数据模式：
"""
        
        for i, pattern in enumerate(patterns):
            prompt += f"{i+1}. {pattern['description']}: 置信度 {pattern['confidence']:.3f}\n"
        
        prompt += f"""
数据统计信息：
- 输出变量范围: [{np.min(task.samples[:, 0]):.3f}, {np.max(task.samples[:, 0]):.3f}]
- 输出变量均值: {np.mean(task.samples[:, 0]):.3f}
- 输出变量标准差: {np.std(task.samples[:, 0]):.3f}

请基于这些数据模式和统计信息，推导出最可能的数学方程。考虑：
1. 数据中的统计模式
2. 变量间的相关性
3. 数据的分布特征
4. 拟合优度

输出格式：直接给出数学表达式，如 y = f(x1, x2, ...)
"""
        
        return prompt
    
    def _parse_equation(self, equation_str: str, task: SEDTask) -> Optional[Dict[str, Any]]:
        """解析和验证方程"""
        try:
            return {
                "expression": equation_str.strip(),
                "confidence": 0.4,
                "program_format": equation_str.strip(),
                "lambda_format": None
            }
        except Exception:
            return None
    
    def generate_hypothesis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成假设"""
        return {
            "type": "experimental",
            "content": data,
            "confidence": data.get("confidence", 0.5)
        }
    
    def evaluate_hypothesis(self, hypothesis: Dict[str, Any], data: Dict[str, Any]) -> float:
        """评估假设"""
        return hypothesis.get("confidence", 0.5)

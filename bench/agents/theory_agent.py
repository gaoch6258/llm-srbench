from typing import List, Dict, Any
import numpy as np
from ..agents.base import BaseAgent, AgentMessage
from ..dataclasses import SEDTask, SearchResult, Equation
from ..searchers.base import BaseSearcher

class TheoryAgent(BaseAgent):
    """理论推导智能体 - 专注于从理论角度分析问题"""
    
    def __init__(self, agent_id: str, name: str, llm_sampler, config):
        super().__init__(agent_id, name)
        self.llm_sampler = llm_sampler
        self.config = config
        self.theory_knowledge_base = self._initialize_theory_knowledge()
    
    def _initialize_theory_knowledge(self) -> Dict[str, Any]:
        """初始化理论知识库"""
        return {
            "physical_principles": [
                "conservation_of_energy", "conservation_of_momentum", 
                "conservation_of_mass", "thermodynamics_laws"
            ],
            "mathematical_patterns": [
                "exponential_growth", "oscillatory_behavior", 
                "polynomial_relationships", "trigonometric_functions"
            ],
            "domain_specific": {
                "physics": ["harmonic_oscillator", "wave_equations", "field_theory"],
                "chemistry": ["reaction_kinetics", "equilibrium", "thermodynamics"],
                "biology": ["population_dynamics", "growth_models", "ecosystem_interactions"]
            }
        }
    
    def process_task(self, task: SEDTask) -> List[SearchResult]:
        """处理任务 - 理论推导"""
        self.state.current_task = task
        
        # 1. 分析问题域和物理背景
        domain_analysis = self._analyze_domain(task)
        
        # 2. 生成理论假设
        hypotheses = self._generate_theoretical_hypotheses(task, domain_analysis)
        
        # 3. 使用LLM进行理论推导
        theoretical_equations = self._derive_equations_theoretically(task, hypotheses)
        
        # 4. 转换为SearchResult格式
        results = []
        for eq_data in theoretical_equations:
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
                    "agent_type": "theory",
                    "confidence": eq_data.get("confidence", 0.5),
                    "reasoning": eq_data.get("reasoning", ""),
                    "hypothesis_id": eq_data.get("hypothesis_id"),
                    "domain_analysis": domain_analysis
                }
            )
            results.append(result)
        
        return results
    
    def _analyze_domain(self, task: SEDTask) -> Dict[str, Any]:
        """分析问题域"""
        domain_hints = []
        
        # 从符号描述中推断域
        for desc in task.symbol_descs:
            desc_lower = desc.lower()
            if any(word in desc_lower for word in ["concentration", "reaction", "kinetics"]):
                domain_hints.append("chemistry")
            elif any(word in desc_lower for word in ["population", "growth", "species"]):
                domain_hints.append("biology")
            elif any(word in desc_lower for word in ["oscillation", "frequency", "amplitude"]):
                domain_hints.append("physics")
        
        return {
            "primary_domain": max(set(domain_hints), key=domain_hints.count) if domain_hints else "general",
            "domain_hints": domain_hints,
            "complexity_level": self._assess_complexity(task)
        }
    
    def _assess_complexity(self, task: SEDTask) -> str:
        """评估问题复杂度"""
        num_vars = len(task.symbols)
        num_samples = len(task.samples)
        
        if num_vars <= 2 and num_samples < 100:
            return "simple"
        elif num_vars <= 4 and num_samples < 500:
            return "medium"
        else:
            return "complex"
    
    def _generate_theoretical_hypotheses(self, task: SEDTask, domain_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成理论假设"""
        hypotheses = []
        
        # 基于域分析生成假设
        domain = domain_analysis["primary_domain"]
        complexity = domain_analysis["complexity_level"]
        
        if domain == "physics":
            hypotheses.extend(self._generate_physics_hypotheses(task, complexity))
        elif domain == "chemistry":
            hypotheses.extend(self._generate_chemistry_hypotheses(task, complexity))
        elif domain == "biology":
            hypotheses.extend(self._generate_biology_hypotheses(task, complexity))
        else:
            hypotheses.extend(self._generate_general_hypotheses(task, complexity))
        
        return hypotheses
    
    def _generate_physics_hypotheses(self, task: SEDTask, complexity: str) -> List[Dict[str, Any]]:
        """生成物理相关假设"""
        hypotheses = []
        
        if complexity == "simple":
            hypotheses.append({
                "type": "linear_relationship",
                "description": "线性关系假设",
                "expected_form": "y = ax + b",
                "confidence": 0.3
            })
            hypotheses.append({
                "type": "quadratic",
                "description": "二次关系假设", 
                "expected_form": "y = ax² + bx + c",
                "confidence": 0.2
            })
        
        if "oscillation" in str(task.symbol_descs).lower():
            hypotheses.append({
                "type": "harmonic_oscillator",
                "description": "简谐振荡假设",
                "expected_form": "y = A*sin(ωt + φ)",
                "confidence": 0.7
            })
        
        return hypotheses
    
    def _generate_chemistry_hypotheses(self, task: SEDTask, complexity: str) -> List[Dict[str, Any]]:
        """生成化学相关假设"""
        hypotheses = []
        
        if "reaction" in str(task.symbol_descs).lower():
            hypotheses.append({
                "type": "first_order_kinetics",
                "description": "一级反应动力学",
                "expected_form": "dA/dt = -k*A",
                "confidence": 0.6
            })
            hypotheses.append({
                "type": "second_order_kinetics", 
                "description": "二级反应动力学",
                "expected_form": "dA/dt = -k*A²",
                "confidence": 0.4
            })
        
        return hypotheses
    
    def _generate_biology_hypotheses(self, task: SEDTask, complexity: str) -> List[Dict[str, Any]]:
        """生成生物相关假设"""
        hypotheses = []
        
        if "population" in str(task.symbol_descs).lower():
            hypotheses.append({
                "type": "exponential_growth",
                "description": "指数增长模型",
                "expected_form": "dP/dt = r*P",
                "confidence": 0.5
            })
            hypotheses.append({
                "type": "logistic_growth",
                "description": "逻辑增长模型",
                "expected_form": "dP/dt = r*P*(1-P/K)",
                "confidence": 0.6
            })
        
        return hypotheses
    
    def _generate_general_hypotheses(self, task: SEDTask, complexity: str) -> List[Dict[str, Any]]:
        """生成通用假设"""
        hypotheses = []
        
        hypotheses.append({
            "type": "polynomial",
            "description": "多项式关系",
            "expected_form": "y = Σ(a_i * x^i)",
            "confidence": 0.3
        })
        
        hypotheses.append({
            "type": "exponential",
            "description": "指数关系",
            "expected_form": "y = a * exp(b*x)",
            "confidence": 0.2
        })
        
        return hypotheses
    
    def _derive_equations_theoretically(self, task: SEDTask, hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用LLM进行理论推导"""
        equations = []
        
        # 构建理论推导提示
        prompt = self._build_theory_prompt(task, hypotheses)
        
        # 使用LLM生成方程
        for i in range(self.config.samples_per_prompt):
            try:
                response = self.llm_sampler.sample_program(prompt)
                equation_str = response[0] if isinstance(response, tuple) else response
                
                # 解析和验证方程
                parsed_eq = self._parse_equation(equation_str, task)
                if parsed_eq:
                    parsed_eq["hypothesis_id"] = f"theory_{i}"
                    parsed_eq["reasoning"] = f"理论推导第{i+1}次尝试"
                    equations.append(parsed_eq)
            except Exception as e:
                print(f"Theory agent error: {e}")
                continue
        
        return equations
    
    def _build_theory_prompt(self, task: SEDTask, hypotheses: List[Dict[str, Any]]) -> str:
        """构建理论推导提示"""
        prompt = f"""
你是一个理论物理学家/数学家，需要从理论角度推导出描述以下现象的数学方程：

问题描述：
- 符号: {task.symbols}
- 符号描述: {task.symbol_descs}
- 符号属性: {task.symbol_properties}

基于理论分析，我提出了以下假设：
"""
        
        for i, hyp in enumerate(hypotheses):
            prompt += f"{i+1}. {hyp['description']}: {hyp['expected_form']} (置信度: {hyp['confidence']})\n"
        
        prompt += f"""
请基于这些理论假设，推导出最可能的数学方程。考虑：
1. 物理/数学原理的适用性
2. 边界条件和初始条件
3. 量纲分析
4. 对称性原理

输出格式：直接给出数学表达式，如 y = f(x1, x2, ...)
"""
        
        return prompt
    
    def _parse_equation(self, equation_str: str, task: SEDTask) -> Optional[Dict[str, Any]]:
        """解析和验证方程"""
        try:
            # 简单的方程解析和验证
            # 这里可以添加更复杂的解析逻辑
            return {
                "expression": equation_str.strip(),
                "confidence": 0.6,  # 理论推导的默认置信度
                "program_format": equation_str.strip(),
                "lambda_format": None  # 需要进一步处理
            }
        except Exception:
            return None
    
    def generate_hypothesis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成假设"""
        return {
            "type": "theoretical",
            "content": data,
            "confidence": 0.5
        }
    
    def evaluate_hypothesis(self, hypothesis: Dict[str, Any], data: Dict[str, Any]) -> float:
        """评估假设"""
        # 基于理论一致性评估
        return hypothesis.get("confidence", 0.5)

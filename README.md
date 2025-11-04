# RM-AgenticAI-LangGraph

## AI-Powered Investment Prospect Analysis System

A sophisticated multi-agent AI system designed for investment advisory firms that provides comprehensive prospect analysis for investment planning using LangGraph and advanced AI/ML integration.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Data Flow](#data-flow)
- [Testing](#testing)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

---

## Overview

### What It Does

RM-AgenticAI-LangGraph is an enterprise-grade system that automates complex financial analysis through an orchestrated multi-agent architecture. It analyzes investment prospects across multiple dimensions (risk, persona, goals, products) to provide relationship managers with data-driven, compliant recommendations.

### Problems It Solves

1. **Prospect Analysis Automation** - Eliminates manual analysis cycles
2. **Holistic Financial Assessment** - Multi-dimensional analysis in unified workflow
3. **AI-Powered Recommendations** - Intelligent, personalized investment guidance
4. **RM Support** - Data-driven insights through interactive interface
5. **Compliance Assurance** - Ensures recommendations meet regulatory requirements

---

## Key Features

- **Multi-Agent Architecture** - 8+ specialized AI agents in orchestrated workflow
- **Hybrid AI Approach** - Combines ML models with LLM-based reasoning
- **Data Quality Validation** - Automatic cleaning and quality assessment
- **Risk Assessment** - Both ML and AI analysis
- **Persona Classification** - Behavioral investor analysis
- **Goal Success Prediction** - Success factors and challenges
- **Product Recommendations** - Intelligent matching with suitability scoring
- **Compliance Checks** - Regulatory validation
- **Interactive Web UI** - Built with Streamlit
- **AI Chat Assistant** - Answers questions about analysis
- **Performance Monitoring** - Agent execution tracking

---

## Architecture

### System Overview

```
User Input (Prospect Data)
           ↓
    ┌─────────────────────────────────────────────┐
    │    LangGraph Workflow Orchestration        │
    └─────────────────────────────────────────────┘
           ↓
    ┌─────────────────────────────────────────────┐
    │  Node 1: Data Analysis                      │
    │  - Validation & Quality Assessment          │
    └─────────────────────────────────────────────┘
           ↓
    ┌─────────────────────────────────────────────┐
    │  Node 2: Risk Assessment                    │
    │  - ML Model + AI Analysis                   │
    └─────────────────────────────────────────────┘
           ↓
    ┌─────────────────────────────────────────────┐
    │  Node 3: Persona Classification             │
    │  - Investor Personality Type                │
    └─────────────────────────────────────────────┘
           ↓
    ┌─────────────────────────────────────────────┐
    │  Node 4: Product Recommendation             │
    │  - Intelligent Filtering & Scoring          │
    └─────────────────────────────────────────────┘
           ↓
    ┌─────────────────────────────────────────────┐
    │  Node 5: Finalization                       │
    │  - Insights & Action Items                  │
    └─────────────────────────────────────────────┘
           ↓
    Comprehensive Analysis Report
```

---

## Project Structure

```
Project/
├── main.py                              # Streamlit web application UI
├── graph.py                             # LangGraph workflow orchestration
├── state.py                             # Pydantic state models
├── tests.py                             # Comprehensive test suite (17 tests)
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
│
├── config/                              # Configuration & Logging
│   ├── __init__.py
│   ├── settings.py                      # Environment & app configuration
│   └── logging_config.py                # Loguru-based logging setup
│
├── langraph_agents/                     # AI Agent Framework
│   ├── __init__.py
│   ├── base_agent.py                    # Abstract base class for agents
│   ├── state_models.py                  # Shared Pydantic models
│   │
│   └── agents/                          # Specialized AI Agents
│       ├── data_analyst_agent.py        # Data validation & quality
│       ├── risk_assessment_agent.py     # Risk profiling (ML + AI)
│       ├── persona_agent.py             # Investor personality classification
│       ├── product_specialist_agent.py  # Product recommendations
│       ├── goal_planning_agent.py       # Goal success prediction
│       ├── compliance_agent.py          # Regulatory compliance
│       ├── meeting_coordinator_agent.py # Meeting scheduling
│       ├── portfolio_optimizer_agent.py # Portfolio optimization
│       └── rm_assistant_agent.py        # Chat assistant
│
├── ml/                                  # Machine Learning Components
│   ├── models/                          # Pre-trained ML Artifacts
│   │   ├── risk_profile_model.pkl
│   │   ├── label_encoders.pkl
│   │   ├── goal_success_model.pkl
│   │   └── goal_success_label_encoders.pkl
│   │
│   └── training/                        # Model Training Pipelines
│       ├── train_models.py              # Orchestrates training
│       ├── predict_risk_profile.py      # Risk model training
│       └── predict_goal_success.py      # Goal model training
│
├── nodes/                               # LangGraph Node Wrappers
│   ├── data_analysis_node.py
│   ├── risk_assessment_node.py
│   ├── persona_node.py
│   ├── product_recommendation_node.py
│   └── finalize_analysis_node.py
│
├── workflow/                            # Workflow Definition
│   └── workflow.py                      # Workflow sequence
│
├── data/                                # Data Inputs
│   └── input_data/
│       ├── prospects.csv                # Sample prospect dataset
│       └── products.csv                 # Product catalog
│
├── utils/                               # Utilities
│   ├── install.py                       # Installation automation
│   ├── quick_fix.py                     # Runtime issue fixes
│   └── retrain_models.py                # Model retraining script
│
└── logs/                                # Runtime Logs
    └── app.log                          # Application log
```

---

## Installation

### Prerequisites

- Python 3.8+
- pip package manager
- Google Gemini API key (free tier available)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs ~20 packages including:
- langraph (Multi-agent framework)
- pydantic (Data validation)
- streamlit (Web UI)
- scikit-learn (ML models)
- google-generativeai (Gemini API)

### Step 2: Configure Environment

Create a `.env` file in the project root:

```bash
# .env file
GEMINI_API_KEY_1=your-google-gemini-api-key-here
LANGCHAIN_API_KEY=optional-langchain-api-key
LANGCHAIN_TRACING_V2=True
LANGCHAIN_PROJECT=rm-agentic-ai
LOG_LEVEL=INFO
DEBUG_MODE=False
```

**Getting your API Key:**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the key to `.env`

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY_1` | - | Google Gemini API key (required) |
| `LOG_LEVEL` | INFO | Logging level: DEBUG, INFO, WARNING, ERROR |
| `DEBUG_MODE` | False | Enable debug output |
| `max_concurrent_agents` | 5 | Maximum parallel agent executions |
| `agent_timeout` | 300 | Timeout per agent (seconds) |
| `cache_ttl` | 3600 | Cache expiration (seconds) |

### Settings (config/settings.py)

```python
from config.settings import get_settings

settings = get_settings()
print(settings.gemini_api_key)  # Access any setting
print(settings.log_level)
```

---

## Usage

### Running the Application

```bash
python3 -m streamlit run main.py
```

**Output:**
```
Collecting usage statistics...
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

**Browser Steps:**
1. Go to http://localhost:8501
2. Select a prospect from sidebar dropdown
3. Click "🚀 Start AI Analysis"
4. View results in tabs: Analysis, Agent Performance, Chat
5. Ask questions in Chat tab

### Running Tests

```bash
python tests.py
```

**Output:**
```
Configuration Tests: 5/5 PASSED
Model Tests: 5/5 PASSED
System Tests: 7/7 PASSED
TOTAL: 17/17 tests PASSED
```

### Training Models

```bash
python ml/training/train_models.py
```

---

## API Reference

### Main Classes

#### ProspectAnalysisWorkflow

Main orchestrator for the workflow.

```python
from graph import ProspectAnalysisWorkflow

workflow = ProspectAnalysisWorkflow()
result = await workflow.analyze_prospect(prospect_data)
```

**Methods:**
- `analyze_prospect(prospect_data)` - Run full workflow
- `get_workflow_state(session_id)` - Get current state
- `get_workflow_summary()` - Get workflow metadata

#### WorkflowState

Complete state container for workflow execution.

```python
from state import WorkflowState, ProspectData

state = WorkflowState(
    workflow_id="uuid",
    prospect=ProspectState(prospect_data=ProspectData(...))
)
```

**Key Attributes:**
- `prospect` - ProspectState with input data
- `analysis` - AnalysisState with risk, persona, goals
- `recommendations` - RecommendationState with products
- `agent_executions` - Agent performance tracking
- `overall_confidence` - Final confidence score

### Agent Base Class

```python
from langraph_agents.base_agent import BaseAgent, CriticalAgent, OptionalAgent

class CustomAgent(CriticalAgent):
    async def execute(self, state: WorkflowState) -> WorkflowState:
        # Implementation
        return state

    def get_prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([...])
```

---

## Data Flow

### Complete Workflow Data Flow

```
INPUT: Prospect Data
    ↓
NODE 1: DATA ANALYSIS
    ├─ Validate required fields
    ├─ Check business logic (age, income ranges)
    ├─ Calculate data_quality_score
    └─ Set: validation_errors, missing_fields
    ↓
NODE 2: RISK ASSESSMENT
    ├─ Load ML model (RandomForestClassifier)
    ├─ Predict risk level (Low/Moderate/High)
    ├─ Generate risk factors via LLM
    └─ Set: risk_level, confidence_score, risk_factors
    ↓
NODE 3: PERSONA CLASSIFICATION
    ├─ Classify persona type via LLM
    ├─ Analyze behavioral patterns
    ├─ Calculate confidence score
    └─ Set: persona_type, characteristics, behavioral_insights
    ↓
NODE 4: PRODUCT RECOMMENDATION
    ├─ Load product catalog
    ├─ Filter by risk, amount, persona
    ├─ Score products by suitability
    ├─ Rank and select top 5
    └─ Set: recommended_products, justification_text
    ↓
NODE 5: FINALIZATION
    ├─ Aggregate confidence scores
    ├─ Generate key insights
    ├─ Generate action items
    └─ Set: overall_confidence, key_insights, action_items
    ↓
OUTPUT: WorkflowState with complete analysis
```

### Key State Dependencies

- `data_quality_score` → Used in Node 2, 5
- `risk_level` → Used in Node 3, 4, 5
- `persona_type` → Used in Node 4, 5
- `suitability_scores` → Used in Node 5

---

## Testing

### Test Suite Overview

**Total: 17 Tests across 3 suites**

#### Configuration Tests (5 tests)
- Environment setup and API keys
- Module imports
- Pydantic settings
- Streamlit compatibility

#### Model Tests (5 tests)
- Model file existence
- Risk model predictions
- Goal model predictions
- Agent integration
- Model performance

#### System Tests (7 tests)
- System imports
- Complete configuration
- Data loading
- Agent initialization
- Workflow creation
- Logging setup
- End-to-end analysis

### Running Tests

```bash
python tests.py
```

### Test Output Example

```
Starting RM-AgenticAI-LangGraph Test Suite...

CONFIGURATION TESTS
  test_environment_setup.....................PASSED
  test_imports...............................PASSED
  test_missing_imports........................PASSED
  test_pydantic_settings......................PASSED
  test_streamlit_compatibility................PASSED
Configuration Tests: 5/5 PASSED

MODEL TESTS
  test_model_files...........................PASSED
  test_risk_model............................PASSED
  test_goal_model............................PASSED
  test_agent_integration.....................PASSED
  test_model_performance.....................PASSED
Model Tests: 5/5 PASSED

SYSTEM TESTS
  test_system_imports........................PASSED
  test_system_configuration..................PASSED
  test_system_data_loading...................PASSED
  test_system_agent_initialization...........PASSED
  test_system_workflow_creation..............PASSED
  test_system_logging........................PASSED
  test_sample_analysis........................PASSED
System Tests: 7/7 PASSED

SUMMARY:
Total Tests: 17
Passed: 17
Failed: 0
Success Rate: 100%
```

---

## Examples

### Example 1: Complete Analysis Output

**Input Prospect:**
```
Name: Rajesh Kumar
Age: 32
Annual Income: ₹1,200,000
Current Savings: ₹500,000
Target Goal: ₹2,000,000 (5 years)
Experience: Intermediate
```

**Output Analysis:**

**Data Quality:** 0.95/1.0 ✅

**Risk Assessment:**
- Level: Moderate
- Confidence: 0.92
- Factors: Medium income, 5-year horizon, intermediate experience

**Persona:** Steady Saver (0.88 confidence)
- Characteristics: Consistent saver, balanced approach, goal-oriented
- Insights: Prefers stable returns, values security

**Goal Success:** Likely (0.72 probability)
- Required monthly investment: ₹30,000
- Success factors: Strong savings base, reasonable horizon
- Challenges: Consistency requirement, market volatility

**Top Recommendations:**
1. Balanced Mutual Fund - 0.94 suitability
2. Fixed Deposit Ladder - 0.87 suitability
3. Equity Index Fund - 0.84 suitability
4. Debt Fund - 0.79 suitability
5. SIP Plan - 0.76 suitability

**Action Items:**
1. Schedule consultation on risk tolerance
2. Create investment plan with allocation
3. Set up ₹30,000 monthly SIP
4. Establish 6-month review schedule
5. Verify emergency fund adequacy

### Example 2: Chat Interaction

**User:** "Why is this prospect classified as Steady Saver?"

**Assistant:** Based on analysis, Rajesh shows steady savings behavior (₹500K accumulated), clear goal definition (5-year target), and intermediate experience level. This combination indicates a goal-oriented, balanced approach characteristic of Steady Saver persona.

**User:** "Which product should be prioritized?"

**Assistant:** Balanced Mutual Fund should be primary (0.94 suitability score). It perfectly matches moderate risk profile with 8-10% expected returns, sufficient for goal achievement. Recommend 40% allocation.

---

## Troubleshooting

### Common Issues

**Issue: "GEMINI_API_KEY_1 not found"**
```
Solution:
1. Create .env file in project root
2. Add: GEMINI_API_KEY_1=your-key
3. Restart application
```

**Issue: "Model files not found"**
```
Solution:
python ml/training/train_models.py
```

**Issue: "Streamlit not found"**
```
Solution:
pip install -r requirements.txt
```

**Issue: "Connection timeout"**
```
Solution:
1. Check internet connection
2. Verify Gemini API quota
3. Check firewall settings
```

### Debug Mode

Enable verbose logging:

```bash
python3 -m streamlit run main.py --logger.level=debug
```

Check logs:

```bash
tail -f logs/app.log
```

---

## Recent Changes

### Resolved Merge Conflicts

All merge conflicts have been resolved in:
- `main.py` - Asyncio event loop handling
- `graph.py` - Workflow orchestration
- `state.py` - State models
- `langraph_agents/agents/goal_planning_agent.py` - Model loading
- `langraph_agents/agents/risk_assessment_agent.py` - Model loading
- `ml/training/predict_goal_success.py` - Training pipeline
- `ml/training/predict_risk_profile.py` - Training pipeline

Latest commit: `3f17ae0 - Resolve merge conflicts from bedffaf`

---

## Contributing

To contribute:

1. Create a feature branch
2. Make changes
3. Run tests: `python tests.py`
4. Submit pull request

---

## License

Proprietary - All Rights Reserved

---

## Support

For issues or questions:
- Check [Troubleshooting](#troubleshooting) section
- Review test output for configuration issues
- Check logs in `logs/app.log`

---

## Project Status

✅ **Production Ready**
- All tests passing (17/17)
- Merge conflicts resolved
- Code quality: High
- Documentation: Complete

Last Updated: 2024

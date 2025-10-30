RM-AgenticAI-LangGraph: AI-Powered Investment Prospect Analysis System

================================================================================
PROBLEM STATEMENT
================================================================================

PROJECT NAME: RM-AgenticAI-LangGraph (Multi-Agent Financial AI System)

WHAT IT DOES:
This is an advanced multi-agent AI system designed for investment advisory firms
that provides comprehensive prospect analysis for investment planning. It is built
using LangGraph (a state management framework for multi-agent systems) and
integrates multiple AI agents with machine learning models.

PROBLEMS IT SOLVES:
1. Prospect Analysis Automation - Automates complex analysis of potential clients
2. Holistic Financial Assessment - Multi-dimensional analysis in unified workflow
3. AI-Powered Recommendations - Intelligent, personalized investment guidance
4. RM Support - Data-driven insights through interactive interface
5. Compliance Assurance - Ensures recommendations meet regulatory requirements

KEY FEATURES:
- Multi-Agent Architecture with 8+ specialized AI agents in orchestrated workflow
- Hybrid AI Approach combining ML models with LLM-based reasoning
- Data Quality Validation with automatic cleaning and quality assessment
- Risk Assessment using both ML and AI analysis
- Persona Classification for behavioral investor analysis
- Goal Success Prediction with success factors and challenges
- Product Recommendations with intelligent matching and suitability scoring
- Compliance Checks for regulatory validation
- Interactive Web UI built with Streamlit
- AI-Powered Chat Assistant for answering analysis questions
- Performance Monitoring with agent execution tracking

================================================================================
FILE STRUCTURE AND DIRECTORY MAP
================================================================================

Project:

main.py                             Entry point - Streamlit web application UI

graph.py                            LangGraph workflow orchestration engine

state.py                            Pydantic state models for workflow

tests.py                            Comprehensive test suite (17 tests)

requirements.txt                    Python dependencies list

config/
    __init__.py
    settings.py                     Application settings and env variables
    logging_config.py               Loguru logging configuration

langraph_agents/
    __init__.py
    base_agent.py                   Abstract base class for all agents
    state_models.py                 Pydantic state models for agents
    agents/
        __init__.py
        data_analyst_agent.py       Data validation and quality checking
        risk_assessment_agent.py     Risk profiling with ML and AI
        persona_agent.py            Investor personality classification
        product_specialist_agent.py Product recommendations generation
        goal_planning_agent.py       Goal success prediction
        compliance_agent.py          Regulatory compliance validation
        meeting_coordinator_agent.py Meeting preparation
        portfolio_optimizer_agent.py Portfolio optimization
        rm_assistant_agent.py        Chat assistant for users

ml/
    __init__.py
    models/
        __init__.py
        risk_profile_model.pkl      Trained risk classifier
        label_encoders.pkl          Encoders for risk model
        goal_success_model.pkl      Trained goal predictor
        goal_success_label_encoders.pkl  Encoders for goal model
    training/
        __init__.py
        train_models.py             Model training orchestration
        predict_risk_profile.py     Risk model training and prediction
        predict_goal_success.py     Goal model training and prediction

nodes/
    __init__.py
    data_analysis_node.py
    risk_assessment_node.py
    persona_node.py
    product_recommendation_node.py
    finalize_analysis_node.py

workflow/
    __init__.py
    workflow.py

data/
    input_data/
        prospects.csv              Sample prospect data
        products.csv               Product catalog

utils/
    __init__.py
    install.py                     Installation utilities
    quick_fix.py                   Quick fixes for common issues
    retrain_models.py              Model retraining script

logs/                              Log files generated at runtime
    app.log
    agents.log

================================================================================
FILE PURPOSES AND IMPORTANT METHODS
================================================================================

SECTION A: CORE APPLICATION FILES

--- main.py (Streamlit Web Application) ---

PURPOSE: Web UI for the entire system, allowing relationship managers to analyze
prospects and interact with the AI system.

IMPORTANT METHODS AND PARAMETERS:

Method: st.set_page_config()
Parameters: title, layout, sidebar_state
Returns: None
Purpose: Configure Streamlit page settings

Method: get_workflow()
Parameters: None
Returns: ProspectAnalysisWorkflow instance
Purpose: Cache and initialize the workflow

Method: check_model_status()
Parameters: None
Returns: Dictionary with model availability status
Purpose: Verify ML models are loaded and available

Method: load_prospects()
Parameters: None
Returns: List of prospects from CSV or dummy data
Purpose: Load prospect data with fallback dummy data

Method: run_analysis(prospect_data)
Parameters: prospect_data (dict with prospect info)
Returns: WorkflowState with complete analysis results
Purpose: Execute workflow analysis synchronously

Method: display_analysis_results(workflow_state)
Parameters: workflow_state (WorkflowState object)
Returns: None
Purpose: Show comprehensive analysis results in UI tabs

Method: generate_chat_response(user_query, analysis_context)
Parameters: user_query (string), analysis_context (dict)
Returns: String response from Gemini LLM
Purpose: Generate AI response to user questions

Method: generate_fallback_response(user_query)
Parameters: user_query (string)
Returns: Rule-based response string
Purpose: Generate response when LLM is unavailable

Method: get_suggested_questions(analysis_results)
Parameters: analysis_results (dict)
Returns: List of 3-5 suggested follow-up questions
Purpose: Generate contextual follow-up question suggestions

Method: display_agent_performance(workflow_state)
Parameters: workflow_state (WorkflowState)
Returns: None
Purpose: Show agent execution metrics and timing

UI SECTIONS:
- Sidebar: Model status, workflow info, analysis settings
- Main Content: Prospect selection, analysis trigger, results
- Tabs: Analysis Results, Agent Performance, Chat Assistant

--- graph.py (LangGraph Workflow Orchestration) ---

PURPOSE: Defines and orchestrates the complete multi-agent workflow with 5 nodes
executing in sequence.

CLASS: ProspectAnalysisWorkflow

Method: __init__()
Parameters: config (Settings instance)
Returns: None
Purpose: Initialize workflow, create agents, build graph

Method: _build_workflow()
Parameters: None
Returns: None
Purpose: Create LangGraph StateGraph with 5 processing nodes

Method: _data_analysis_node(state)
Parameters: state (WorkflowState)
Returns: Updated WorkflowState
Purpose: Node 1 - Data validation and quality assessment
Validates: Required fields, business logic, valid ranges
Sets: data_quality_score, validation_errors

Method: _risk_assessment_node(state)
Parameters: state (WorkflowState)
Returns: Updated WorkflowState
Purpose: Node 2 - ML-based risk profiling plus AI analysis
Uses: ML model and LLM
Sets: risk_level, confidence_score, risk_factors

Method: _persona_classification_node(state)
Parameters: state (WorkflowState)
Returns: Updated WorkflowState
Purpose: Node 3 - Investor personality classification
Classifies: Aggressive Growth, Steady Saver, Cautious Planner
Sets: persona_type, confidence_score, behavioral_insights

Method: _product_recommendation_node(state)
Parameters: state (WorkflowState)
Returns: Updated WorkflowState
Purpose: Node 4 - Intelligent product recommendations
Returns: Top 5 products with suitability scores
Sets: recommended_products[], justification_text

Method: _finalize_analysis_node(state)
Parameters: state (WorkflowState)
Returns: Updated WorkflowState
Purpose: Node 5 - Summary generation and insights
Sets: overall_confidence, key_insights[], action_items[]

Method: _generate_key_insights()
Parameters: None
Returns: List of insight strings
Purpose: Extract main findings from complete analysis

Method: _generate_action_items()
Parameters: None
Returns: List of action item strings
Purpose: Generate RM action items from analysis

Method: analyze_prospect(prospect_data)
Parameters: prospect_data (ProspectData object)
Returns: Awaitable that returns WorkflowState
Purpose: Main async entry point to run full workflow

Method: get_workflow_state()
Parameters: None
Returns: Current WorkflowState
Purpose: Retrieve current workflow execution state

WORKFLOW EXECUTION ORDER:
data_analysis → risk_assessment → persona_classification →
product_recommendation → finalize_analysis → COMPLETE

--- state.py (Pydantic State Models) ---

PURPOSE: Define all data models for workflow state and validation.

KEY CLASSES:

Class: ProspectData
Fields:
  prospect_id (string): Unique identifier
  name (string): Prospect name
  age (integer): Age in years
  annual_income (float): Annual income amount
  current_savings (float): Current savings amount
  target_goal_amount (float): Investment target
  investment_horizon_years (integer): Years to investment goal
  number_of_dependents (integer): Count of dependents
  investment_experience_level (enum): Beginner/Intermediate/Advanced
  investment_goal (string): Goal description

Class: RiskAssessmentResult
Fields:
  risk_level (enum): Low, Moderate, or High
  confidence_score (float): 0.0-1.0 confidence
  risk_factors (list): Identified risk factors
  recommendations (list): Risk mitigation recommendations

Class: GoalPredictionResult
Fields:
  goal_success (string): Likely or Unlikely
  probability (float): 0.0-1.0 success probability
  success_factors (list): Factors supporting success
  challenges (list): Potential challenges
  timeline_analysis (dict): Analysis of timeline feasibility

Class: PersonaResult
Fields:
  persona_type (string): Classification type
  confidence_score (float): 0.0-1.0 confidence
  characteristics (list): Persona characteristics
  behavioral_insights (list): Behavioral analysis insights

Class: ProductRecommendation
Fields:
  product_id (string): Product identifier
  product_name (string): Product name
  product_type (string): Type of product
  suitability_score (float): 0.0-1.0 match score
  justification (string): Why recommended
  risk_alignment (string): How risk aligns
  expected_returns (string): Expected return range
  fees (string): Fee structure

Class: ComplianceCheck
Fields:
  is_compliant (boolean): Meets all requirements
  compliance_score (float): 0.0-1.0
  violations (list): Regulatory violations found
  warnings (list): Warnings or concerns
  required_disclosures (list): Required disclosures

Class: WorkflowState (Main state container)
Fields:
  workflow_id (string): Unique workflow execution ID
  prospect (ProspectState): Prospect data and validation
  analysis (AnalysisState): Risk, persona, goals
  recommendations (RecommendationState): Product recommendations
  chat (ChatState): Chat history
  agent_executions (list): Agent execution tracking
  timestamps (dict): Execution times
  overall_confidence (float): Final confidence score

Method: add_agent_execution(agent_name, start_time)
Parameters: agent_name (string), start_time (datetime)
Returns: None
Purpose: Track when agent starts execution

Method: complete_agent_execution(agent_name, status, end_time, error_message)
Parameters: agent_name (string), status (string), end_time (datetime),
           error_message (string, optional)
Returns: None
Purpose: Mark agent as complete with status and metrics

Method: get_execution_summary()
Parameters: None
Returns: Dictionary with execution statistics
Purpose: Get performance metrics across all agents


SECTION B: CONFIGURATION FILES

--- config/settings.py (Settings Management) ---

PURPOSE: Centralized configuration using Pydantic Settings for validation.

CLASS: Settings

CONFIGURATION FIELDS:

API KEYS:
  gemini_api_key (string): Google Gemini API key
  langchain_api_key (string): LangChain API key

LANGSMITH CONFIG:
  langchain_tracing_v2 (boolean): Enable tracing
  langchain_project (string): Project name

APPLICATION SETTINGS:
  log_level (string): Logging level (INFO, DEBUG, ERROR)
  enable_monitoring (boolean): Enable performance monitoring
  debug_mode (boolean): Enable debug output

PERFORMANCE:
  max_concurrent_agents (integer): Max concurrent executions (default 5)
  agent_timeout (integer): Timeout in seconds per agent (default 300)
  cache_ttl (integer): Cache expiration in seconds (default 3600)

FILE PATHS:
  data_dir (string): Data directory path
  models_dir (string): Models directory path
  output_dir (string): Output directory path

MODEL PATHS:
  risk_model_path (string): Risk classifier model location
  goal_model_path (string): Goal predictor model location
  risk_encoders_path (string): Risk model encoders location
  goal_encoders_path (string): Goal model encoders location

DATA FILES:
  prospects_csv (string): Prospects data file path
  products_csv (string): Products catalog file path

AGENT CONFIG:
  default_temperature (float): LLM temperature (default 0.1)
  max_tokens (integer): Max output tokens (default 4000)

Function: get_settings()
Parameters: None
Returns: Settings instance
Purpose: Get settings object with loaded configuration

Function: get_cached_settings()
Parameters: None
Returns: Settings instance (from cache if available)
Purpose: Get cached settings for performance

--- config/logging_config.py (Logging Setup) ---

PURPOSE: Configure structured logging with loguru.

Function: setup_logging(log_level)
Parameters: log_level (string): DEBUG, INFO, WARNING, ERROR
Returns: Logger instance
Purpose: Initialize console and file logging

Logging Configuration:
- Console: Colored output with timestamp, level, module, message
- File (app.log): Rotation at 10MB, retention 30 days
- File (agents.log): Rotation at 5MB, retention 7 days

Function: get_logger(name)
Parameters: name (string): Module name for logger
Returns: Logger instance
Purpose: Get logger instance for specific module


SECTION C: AGENT FILES (Individual Agent Implementations)

--- langraph_agents/base_agent.py (Base Class) ---

PURPOSE: Abstract base class providing common functionality for all agents.

ABSTRACT CLASS: BaseAgent

Method: __init__(name, description, llm, temperature, max_tokens)
Parameters:
  name (string): Agent identifier
  description (string): Agent description
  llm (object): Language model instance
  temperature (float): LLM creativity parameter
  max_tokens (integer): Max output token limit
Returns: None
Purpose: Initialize agent with configuration

Method: execute(state)
Parameters: state (WorkflowState): Current workflow state
Returns: Updated WorkflowState
Purpose: Main abstract execution method (implemented by subclasses)

Method: run(state)
Parameters: state (WorkflowState)
Returns: Updated WorkflowState
Purpose: Execute with error handling and monitoring

Method: validate_input(state)
Parameters: state (WorkflowState)
Returns: Boolean or raises ValidationError
Purpose: Validate input state before execution

Method: validate_output(state)
Parameters: state (WorkflowState)
Returns: Boolean or raises ValidationError
Purpose: Validate output state after execution

Method: is_critical()
Parameters: None
Returns: Boolean
Purpose: Return True if agent must succeed for workflow

Method: generate_response(prompt)
Parameters: prompt (string): Prompt for LLM
Returns: String response from LLM
Purpose: Call LLM and return response

Method: get_system_prompt()
Parameters: None
Returns: String system prompt
Purpose: Return system prompt for agent context

Method: get_performance_metrics()
Parameters: None
Returns: Dictionary with execution statistics
Purpose: Return execution time and performance data

Method: reset_metrics()
Parameters: None
Returns: None
Purpose: Clear collected performance statistics

SUBCLASSES:
- CriticalAgent: Raises exception on failure
- OptionalAgent: Returns state on failure, continues workflow

--- langraph_agents/agents/data_analyst_agent.py ---

PURPOSE: Validate prospect data quality and perform data cleaning.

CLASS: DataAnalystAgent(CriticalAgent)

CRITICAL: Yes - Workflow stops if validation fails

Method: execute(state)
Parameters: state (WorkflowState)
Returns: Updated WorkflowState
Purpose: Main execution - validate and clean data

Method: _validate_data_quality(prospect_data)
Parameters: prospect_data (ProspectData)
Returns: Dictionary with validation results and quality score
Purpose: Check required fields, business logic, valid ranges

Method: _clean_and_enhance_data(prospect_data)
Parameters: prospect_data (ProspectData)
Returns: Cleaned ProspectData
Purpose: AI-assisted data correction and enhancement

Method: validate_input(state)
Parameters: state (WorkflowState)
Returns: Boolean
Purpose: Check prospect data exists

Method: validate_output(state)
Parameters: state (WorkflowState)
Returns: Boolean
Purpose: Check data quality score is set

VALIDATION CHECKS PERFORMED:
- Required fields present (name, age, income, etc.)
- Age range: 18-100 years
- Annual income: minimum 50,000
- Savings amount: non-negative
- Target goal greater than current savings
- Investment horizon greater than 0
- Experience level valid: Beginner/Intermediate/Advanced

DATA QUALITY SCORE: 0.0-1.0 based on completeness and validity

--- langraph_agents/agents/risk_assessment_agent.py ---

PURPOSE: Assess investment risk profile using ML models and AI analysis.

CLASS: RiskAssessmentAgent(CriticalAgent)

CRITICAL: Yes

Method: execute(state)
Parameters: state (WorkflowState)
Returns: Updated WorkflowState with risk assessment
Purpose: Main execution combining ML and AI approaches

Method: _load_models()
Parameters: None
Returns: Tuple (model, label_encoders) or (None, None) if missing
Purpose: Load ML risk model and encoders

Method: _ml_risk_assessment(prospect_data)
Parameters: prospect_data (ProspectData)
Returns: String risk_level from ML prediction
Purpose: Use trained RandomForest model for prediction

Method: _rule_based_risk_assessment(prospect_data)
Parameters: prospect_data (ProspectData)
Returns: String risk_level from rules
Purpose: Fallback scoring algorithm when ML model unavailable

SCORING ALGORITHM:
  - Age factor: Young (2 pts) to Old (0 pts)
  - Income factor: High income (+2 pts)
  - Investment horizon: Long (+2 pts)
  - Experience level: Advanced (+2 pts)
  - Dependents: More dependents (-1 pt)
  - Score threshold: >=6=High, >=3=Moderate, else=Low

Method: _ai_risk_analysis(state)
Parameters: state (WorkflowState)
Returns: Dictionary with factors and recommendations
Purpose: LLM generates risk factors and recommendations

Method: get_prompt_template()
Parameters: None
Returns: String template for risk analysis prompt
Purpose: Return prompt structure for AI analysis

RISK LEVELS: Low, Moderate, High

CONFIDENCE SCORE: 0.0-1.0

--- langraph_agents/agents/persona_agent.py ---

PURPOSE: Classify investor personality and behavioral patterns.

CLASS: PersonaAgent(OptionalAgent)

CRITICAL: No - Workflow continues with default if fails

PERSONA TYPES:
  Aggressive Growth: High risk, long horizon, growth-focused
  Steady Saver: Balanced, consistent, goal-oriented
  Cautious Planner: Conservative, capital preservation, low risk

Method: execute(state)
Parameters: state (WorkflowState)
Returns: Updated WorkflowState
Purpose: Classify persona and generate insights

Method: _classify_persona(prospect_data, risk_assessment)
Parameters: prospect_data (ProspectData), risk_assessment (RiskAssessmentResult)
Returns: String persona type
Purpose: LLM classification using persona descriptions

Method: _generate_behavioral_insights(persona_type)
Parameters: persona_type (string)
Returns: List of behavioral insight strings
Purpose: AI-generated behavioral analysis specific to persona

Method: _extract_persona_type(llm_response)
Parameters: llm_response (string)
Returns: String persona type or default
Purpose: Parse LLM response for persona type

Method: _calculate_confidence_score(prospect_data, persona_type)
Parameters: prospect_data (ProspectData), persona_type (string)
Returns: Float 0.0-1.0 confidence
Purpose: Score based on data alignment with persona

CONFIDENCE CALCULATION:
  - Age alignment: ±0.2
  - Investment horizon: ±0.15
  - Experience level: ±0.1
  - Income alignment: ±0.05

Method: get_classification_prompt()
Parameters: None
Returns: String prompt for classification
Purpose: Return prompt for persona classification

Method: get_insights_prompt()
Parameters: None
Returns: String prompt for behavioral insights
Purpose: Return prompt for insights generation

--- langraph_agents/agents/product_specialist_agent.py ---

PURPOSE: Generate intelligent product recommendations based on prospect profile.

CLASS: ProductSpecialistAgent(CriticalAgent)

CRITICAL: Yes

Method: execute(state)
Parameters: state (WorkflowState)
Returns: Updated WorkflowState with recommendations
Purpose: Generate recommendations and justifications

Method: _load_products()
Parameters: None
Returns: List of ProductRecommendation objects
Purpose: Load products from CSV or create dummy products

Method: _create_dummy_products()
Parameters: None
Returns: List of 10-15 sample products
Purpose: Create test products when CSV unavailable

Method: _filter_products(products, prospect_data, risk_assessment, persona)
Parameters: products (list), prospect_data (ProspectData),
           risk_assessment (RiskAssessmentResult), persona (string)
Returns: Filtered list of eligible products
Purpose: Filter by risk level, investment amount, and persona

FILTERING RULES:
  - Risk mapping: Low→[Low], Moderate→[Low,Moderate], High→[All]
  - Max investment: 80% of savings or 500,000 rupees
  - Persona filtering: Aggressive→High risk products, Cautious→Low risk

Method: _generate_recommendations(filtered_products, state)
Parameters: filtered_products (list), state (WorkflowState)
Returns: Sorted list of top 5 ProductRecommendation objects
Purpose: Score and rank filtered products

Method: _calculate_suitability_score(product, state)
Parameters: product (dict), state (WorkflowState)
Returns: Float 0.0-1.0 suitability score
Purpose: Calculate product suitability

SCORING COMPONENTS:
  - Risk alignment: +0.3
  - Investment amount alignment: +0.1
  - Persona alignment: +0.1

Method: _generate_product_justification(product, state)
Parameters: product (dict), state (WorkflowState)
Returns: String justification for product
Purpose: LLM-generated justification per product

Method: _generate_justification(recommended_products, state)
Parameters: recommended_products (list), state (WorkflowState)
Returns: String overall recommendation justification
Purpose: Overall justification for all recommendations

Method: validate_input(state)
Parameters: state (WorkflowState)
Returns: Boolean
Purpose: Check data and risk assessment exist

Method: validate_output(state)
Parameters: state (WorkflowState)
Returns: Boolean
Purpose: Check recommendations and justification set

OUTPUT: Top 5 products sorted by suitability score

--- langraph_agents/agents/goal_planning_agent.py ---

PURPOSE: Predict investment goal success and feasibility analysis.

CLASS: GoalPlanningAgent(CriticalAgent)

Method: execute(state)
Parameters: state (WorkflowState)
Returns: Updated WorkflowState
Purpose: Predict goal success and analyze feasibility

Method: _ml_goal_prediction(prospect_data)
Parameters: prospect_data (ProspectData)
Returns: Dictionary with prediction and probability
Purpose: Use trained goal success model

Method: _rule_based_goal_prediction(prospect_data)
Parameters: prospect_data (ProspectData)
Returns: Dictionary with prediction and probability
Purpose: Fallback algorithm for goal prediction

RULE-BASED ALGORITHM:
  - Calculate required monthly investment
  - Determine affordable investment (20% of monthly income)
  - Match investment ratio to success probability
  - Success: Likely (prob>=0.6) or Unlikely

Method: _ai_goal_analysis(prospect_data, prediction)
Parameters: prospect_data (ProspectData), prediction (dict)
Returns: Dictionary with success factors, challenges, timeline
Purpose: LLM analyzes feasibility and barriers

Method: _parse_goal_analysis(llm_response)
Parameters: llm_response (string)
Returns: Dictionary with parsed success factors, challenges, timeline
Purpose: Extract structured data from LLM response

Method: get_prompt_template()
Parameters: None
Returns: String prompt template
Purpose: Return prompt for goal analysis

OUTPUT FIELDS:
  - goal_success: Likely or Unlikely
  - probability: 0.0-1.0
  - success_factors: List of enabling factors
  - challenges: List of potential challenges
  - timeline_analysis: Analysis of feasibility timeline

--- langraph_agents/agents/compliance_agent.py ---

PURPOSE: Validate recommendations against regulatory compliance rules.

CLASS: ComplianceAgent(CriticalAgent)

COMPLIANCE RULES CHECKED:
  1. Max single product allocation: 60%
  2. Min diversification: 2+ products required
  3. High-risk age limit: Must be under 65 years
  4. Max investment to income ratio: 30%
  5. Min emergency fund: 6 months of expenses

Method: execute(state)
Parameters: state (WorkflowState)
Returns: Updated WorkflowState
Purpose: Perform comprehensive compliance checks

Method: _perform_compliance_checks(state)
Parameters: state (WorkflowState)
Returns: Dictionary with check results
Purpose: Check all 5 compliance rules

Method: _calculate_compliance_score(violations, warnings)
Parameters: violations (list), warnings (list)
Returns: Float 0.0-1.0 compliance score
Purpose: Calculate score with penalty system

SCORING ALGORITHM:
  - Base score: 1.0
  - Per violation: -0.3
  - Per warning: -0.1
  - Minimum: 0.0

Method: _generate_required_disclosures(violations, warnings)
Parameters: violations (list), warnings (list)
Returns: List of required disclosure strings
Purpose: Generate regulatory disclosure texts


SECTION D: MACHINE LEARNING FILES

--- ml/training/train_models.py ---

PURPOSE: Orchestrate all ML model training and saving.

Function: main()
Parameters: None
Returns: Boolean (True if success, False if error)
Purpose: Main entry point for model training

Execution Flow:
  1. Calls train_risk_model()
  2. Calls train_goal_model()
  3. Returns overall success status

--- ml/training/predict_risk_profile.py ---

PURPOSE: Train and use risk profile prediction model.

Function: train_risk_model()
Parameters: None
Returns: Boolean (True if model exists/trained, False if failed)
Purpose: Load or train risk profile model

Function: predict_risk_profile(prospect_data)
Parameters: prospect_data (ProspectData or dict)
Returns: Awaitable returning (risk_level, confidence_score)
Purpose: Predict risk level using ML model

MODEL DETAILS:
  Type: sklearn RandomForestClassifier
  Input Features: 7 features
    - age (integer)
    - annual_income (float)
    - current_savings (float)
    - target_goal_amount (float)
    - investment_horizon_years (integer)
    - number_of_dependents (integer)
    - investment_experience_level (encoded: 0-2)
  Output Classes: 3 classes
    - 0 = Low risk
    - 1 = Moderate risk
    - 2 = High risk
  Label Encoders: Stored in label_encoders.pkl for categorical encoding

--- ml/training/predict_goal_success.py ---

PURPOSE: Train and use goal success prediction model.

Function: train_goal_model()
Parameters: None
Returns: Boolean (True if success, False if error)
Purpose: Load or train goal success model

Function: predict_goal_success(prospect_data)
Parameters: prospect_data (ProspectData or dict)
Returns: Awaitable returning (prediction, probability)
Purpose: Predict goal success likelihood

MODEL DETAILS:
  Type: sklearn RandomForestClassifier or Regressor
  Input Features: 6 features
    - age (integer)
    - annual_income (float)
    - current_savings (float)
    - target_goal_amount (float)
    - investment_experience_level (encoded)
    - investment_horizon_years (integer)
  Output: Binary classification or continuous probability
    - 0 = Unlikely to achieve goal
    - 1 = Likely to achieve goal
  Label Encoders: Stored in goal_success_label_encoders.pkl

--- tests.py (Test Suite) ---

PURPOSE: Comprehensive testing covering configuration, models, and system integration.

Total Tests: 17 tests

TEST SUITE 1: Configuration Tests (5 tests)

Test: test_environment_setup()
Purpose: Verify .env file and API keys are configured
Checks: GEMINI_API_KEY_1 present and valid

Test: test_imports()
Purpose: Verify critical module imports work
Imports tested: streamlit, langchain, langgraph, pydantic

Test: test_missing_imports()
Purpose: Test module resolution and error handling
Verifies: Proper import exception handling

Test: test_pydantic_settings()
Purpose: Verify settings loading from environment
Checks: Settings instantiation and field validation

Test: test_streamlit_compatibility()
Purpose: Test Streamlit integration
Verifies: Streamlit session state and components

TEST SUITE 2: Model Tests (5 tests)

Test: test_model_files()
Purpose: Check ML model files exist and load correctly
Checks: risk_profile_model.pkl, goal_success_model.pkl existence

Test: test_risk_model()
Purpose: Test risk assessment model predictions
Verifies: Model produces valid risk level output

Test: test_goal_model()
Purpose: Test goal success model predictions
Verifies: Model produces valid probability output

Test: test_agent_integration()
Purpose: Test agents load models correctly
Verifies: Agent model loading without errors

Test: test_model_performance()
Purpose: Test model performance with sample data
Verifies: Models handle multiple samples correctly

TEST SUITE 3: System Tests (7 tests)

Test: test_system_imports()
Parameters: None
Purpose: Verify all system-level imports
Checks: Core modules, agents, workflow imports

Test: test_system_configuration()
Purpose: Test complete configuration loading
Verifies: Settings, logging, environment setup

Test: test_system_data_loading()
Purpose: Test data file loading
Checks: prospects.csv, products.csv loading

Test: test_system_agent_initialization()
Purpose: Test all agents instantiate correctly
Verifies: DataAnalystAgent, RiskAssessmentAgent, PersonaAgent,
         ProductSpecialistAgent initialization

Test: test_system_workflow_creation()
Purpose: Test workflow initialization
Verifies: ProspectAnalysisWorkflow creation and state setup

Test: test_system_logging()
Purpose: Test logging configuration
Verifies: Loguru setup, console and file logging

Test: test_sample_analysis()
Purpose: End-to-end analysis with sample prospect
Verifies: Complete workflow execution from start to finish
Uses async execution for full integration test

RUN COMMAND: python tests.py

OUTPUT EXAMPLE:
[✓] Configuration Tests Passed: 5/5
[✓] Model Tests Passed: 5/5
[✓] System Tests Passed: 7/5
[✓] TOTAL: 17/17 tests passed


SECTION E: KEY CONFIGURATION AND DATA

Environment Variables (.env file):

GEMINI_API_KEY_1=<Your-Google-Gemini-API-Key-Here>
LANGCHAIN_API_KEY=<Optional-LangChain-API-Key>
LANGCHAIN_TRACING_V2=True
LANGCHAIN_PROJECT=rm-agentic-ai
LOG_LEVEL=INFO
DEBUG_MODE=False

Key Configuration Settings (from config/settings.py):

max_concurrent_agents=5         Maximum parallel agent executions
agent_timeout=300               Timeout per agent in seconds
cache_ttl=3600                  Cache expiration in seconds
default_temperature=0.1         LLM creativity (low for consistency)
max_tokens=4000                 Maximum LLM output tokens

================================================================================
DATA FLOW THROUGH THE SYSTEM
================================================================================

COMPLETE WORKFLOW DATA FLOW:

INPUT STAGE:
  Prospect data comes from:
    - CSV file (prospects.csv)
    - Streamlit form input
    - API input (if extended)

DATA STRUCTURE AT INPUT:
  ProspectData object containing:
    - prospect_id, name, age
    - annual_income, current_savings
    - target_goal_amount, investment_horizon_years
    - number_of_dependents
    - investment_experience_level
    - investment_goal

WORKFLOW STAGE 1: DATA ANALYSIS NODE

INPUT: Raw ProspectData
PROCESSING:
  - DataAnalystAgent validates all required fields
  - Checks business logic (age 18-100, income >= 50K, etc.)
  - Calculates data_quality_score (0-1)
  - Identifies validation_errors and missing_fields
OUTPUT STATE UPDATES:
  - prospect.data_quality_score = float (0-1)
  - prospect.validation_errors = list of error strings
  - prospect.missing_fields = list of field names

WORKFLOW STAGE 2: RISK ASSESSMENT NODE

INPUT: Validated ProspectData + data_quality_score
PROCESSING:
  - RiskAssessmentAgent loads ML model (if available)
  - Runs ML prediction with 7 input features
  - Falls back to rule-based scoring if model unavailable
  - Calls LLM for risk factor analysis
OUTPUT STATE UPDATES:
  - analysis.risk_assessment.risk_level = Low/Moderate/High
  - analysis.risk_assessment.confidence_score = float (0-1)
  - analysis.risk_assessment.risk_factors = list of strings
  - analysis.risk_assessment.recommendations = list of strings

WORKFLOW STAGE 3: PERSONA CLASSIFICATION NODE

INPUT: ProspectData + risk_assessment results
PROCESSING:
  - PersonaAgent sends prospect profile + risk to LLM
  - LLM classifies into: Aggressive Growth / Steady Saver / Cautious Planner
  - Calculates confidence based on data alignment
  - Generates behavioral insights
OUTPUT STATE UPDATES:
  - analysis.persona_classification.persona_type = string
  - analysis.persona_classification.confidence_score = float (0-1)
  - analysis.persona_classification.characteristics = list
  - analysis.persona_classification.behavioral_insights = list

WORKFLOW STAGE 4: PRODUCT RECOMMENDATION NODE

INPUT: ProspectData + risk_assessment + persona_classification + product_catalog
PROCESSING:
  - ProductSpecialistAgent loads product catalog (CSV or dummy data)
  - Filters products by:
    - Risk level alignment (Low-risk investors → Low risk products)
    - Available investment amount (80% of savings, max 500K rupees)
    - Persona type alignment
  - Scores remaining products by suitability (risk + amount + persona alignment)
  - Ranks and selects top 5 products
  - Generates AI justification for each product
OUTPUT STATE UPDATES:
  - recommendations.recommended_products = list of 5 ProductRecommendation objects
    Each product contains:
      - product_id, product_name, product_type
      - suitability_score, justification
      - risk_alignment, expected_returns, fees
  - recommendations.justification_text = overall justification string

WORKFLOW STAGE 5: FINALIZE ANALYSIS NODE

INPUT: All state from nodes 1-4
PROCESSING:
  - Aggregates confidence scores from all agents
  - Generates key_insights (main findings)
  - Generates action_items (RM to-do list)
  - Records execution timestamps
  - Calculates overall_confidence (average of component scores)
OUTPUT STATE UPDATES:
  - overall_confidence = float (0-1, average of all scores)
  - key_insights = list of insight strings
  - action_items = list of RM action items
  - workflow_completion_time = timestamp
  - workflow_status = COMPLETED / FAILED / PARTIAL

FINAL OUTPUT TO UI:

WorkflowState object containing complete analysis:
  - ProspectState: Original data + validation results
  - AnalysisState: Risk, persona, goals analyses
  - RecommendationState: Products and justifications
  - ChatState: Empty initially, filled by user questions
  - Agent execution tracking: Timing and status for each agent
  - Overall metrics: Confidence, insights, action items

STATE DEPENDENCY CHAIN:

data_quality_score (Node 1)
  └─ Used by Node 2: Skip analysis if too low
  └─ Used by Node 5: Include in final confidence

risk_level (Node 2)
  └─ Used by Node 3: Consider in persona classification
  └─ Used by Node 4: Filter products by risk level
  └─ Used by Node 5: Include in confidence calculation

persona_type (Node 3)
  └─ Used by Node 4: Filter and score products by persona
  └─ Used by Node 5: Generate persona-based action items

suitability_scores (Node 4)
  └─ Used by Node 5: Rank importance of recommendations
  └─ Used by Node 5: Include in final confidence

================================================================================
HOW TO RUN THE PROJECT
================================================================================

SETUP PHASE:

Step 1 - Install Dependencies
Command: pip install -r requirements.txt
Output: Downloads and installs all packages from requirements.txt
        Should complete without errors if system is compatible
        Installs approximately 15-20 packages

Step 2 - Configure Environment
Command: Create .env file in root directory
Content: Add GEMINI_API_KEY_1=your-google-api-key-here
         (Get API key from: https://makersuite.google.com/app/apikey)
Output: .env file created in root directory

RUNNING THE APPLICATION:

Step 3 - Start Streamlit Application
Command: python3 -m streamlit run main.py
Output Examples:
  Collecting usage statistics...
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
  Network URL: http://x.x.x.x:8501

Step 4 - Use the Application
Browser URL: http://localhost:8501
UI Elements:
  - Sidebar: Model status, settings, analysis info
  - Prospect selector: Choose from loaded prospects
  - Analysis button: Triggers workflow
  - Results tabs: Analysis, Performance, Chat
  - Chat interface: Ask questions about results

ALTERNATIVE RUNNING OPTIONS:

Run Tests:
Command: python tests.py
Output: Test results with pass/fail summary
        17 tests total across 3 suites
        Should show: [✓] ALL TESTS PASSED

Train Models:
Command: python ml/training/train_models.py
Output: Model training progress and status
        Saves models to ml/models/ directory
        Should show: Successfully trained and saved models

Retrain Models:
Command: python utils/retrain_models.py
Output: Retrains existing models with fresh data
        Updates model files in place

Debug Mode:
Command: python3 -m streamlit run main.py --logger.level=debug
Output: Verbose logging showing all operations
        Useful for troubleshooting

Specific Configuration:
Command: python3 -m streamlit run main.py --config.toml (if config exists)
Output: Uses custom configuration from config.toml file

================================================================================
EXPECTED OUTPUT EXAMPLES
================================================================================

EXAMPLE 1: SUCCESSFUL ANALYSIS OUTPUT

When user selects a prospect and runs analysis, output includes:

Prospect: Rajesh Kumar (Age 32)
Annual Income: 1,200,000 INR
Current Savings: 500,000 INR
Target Goal: 2,000,000 INR (5 years)
Experience: Intermediate

DATA QUALITY ASSESSMENT:
Data Quality Score: 0.95/1.0
Status: Excellent
Issues: None

RISK ASSESSMENT:
Risk Level: Moderate
Confidence: 0.92
Risk Factors:
  - Medium income level
  - 5-year investment horizon (moderate)
  - Intermediate experience level
Recommendations:
  - Diversify across asset classes
  - Consider debt instruments for stability
  - Regular portfolio review recommended

PERSONA CLASSIFICATION:
Investor Type: Steady Saver
Confidence: 0.88
Characteristics:
  - Consistent saving behavior
  - Balanced risk approach
  - Goal-oriented mindset
Behavioral Insights:
  - Prefers stable, predictable returns
  - Values financial security
  - Open to long-term commitments

GOAL SUCCESS ANALYSIS:
Success Likelihood: Likely
Probability: 0.72
Required Monthly Investment: 30,000 INR
Success Factors:
  - Strong current savings base
  - Reasonable investment horizon
  - Stable income source
Challenges:
  - Requires consistent monthly commitment
  - Market volatility impact
Timeline: Feasible within 5-year window

PRODUCT RECOMMENDATIONS:
Product 1: Balanced Mutual Fund
Suitability Score: 0.94
Expected Returns: 8-10% annually
Justification: Aligns with moderate risk profile and savings goal

Product 2: Fixed Deposit Ladder
Suitability Score: 0.87
Expected Returns: 6-7% annually
Justification: Provides stability and predictable returns

Product 3: Equity Index Fund
Suitability Score: 0.84
Expected Returns: 10-12% annually
Justification: Growth component with diversification

Product 4: Debt Fund
Suitability Score: 0.79
Expected Returns: 5-6% annually
Justification: Capital preservation component

Product 5: Systematic Investment Plan (SIP)
Suitability Score: 0.76
Expected Returns: 9-11% annually
Justification: Disciplined approach aligned with steady saver persona

KEY INSIGHTS:
- Prospect is well-positioned for goal achievement
- Balanced approach suits risk profile and experience
- Strong savings discipline evident
- Timeline realistic for goal achievement

ACTION ITEMS FOR RELATIONSHIP MANAGER:
1. Schedule initial consultation to discuss risk tolerance in detail
2. Create investment plan with recommended product allocation
3. Set up monthly SIP contribution of 30,000 INR
4. Establish 6-month review schedule
5. Discuss emergency fund adequacy (current: good)

AGENT PERFORMANCE METRICS:
DataAnalystAgent: Execution time 0.23s, Status: Success
RiskAssessmentAgent: Execution time 0.45s, Status: Success
PersonaAgent: Execution time 0.38s, Status: Success
ProductSpecialistAgent: Execution time 0.62s, Status: Success
FinalizationAgent: Execution time 0.19s, Status: Success

Total Workflow Time: 1.87 seconds
Overall Confidence Score: 0.89/1.0
Workflow Status: COMPLETED SUCCESSFULLY


EXAMPLE 2: CHAT INTERACTION OUTPUT

User Question: Why is this prospect classified as Steady Saver?

AI Response: Based on the analysis, Rajesh Kumar is classified as a Steady
Saver because of several key indicators. First, he has accumulated 500,000 INR
in savings over time, showing consistent financial discipline. Second, his
investment horizon of 5 years with a clear financial goal demonstrates
goal-oriented planning. Third, his intermediate experience level suggests a
balanced approach rather than aggressive speculation. This combination of
steady savings behavior, clear goal definition, and moderate experience places
him squarely in the Steady Saver category, which aligns well with balanced
mutual funds and systematic investment plans.

User Question: Which product should he prioritize?

AI Response: The Balanced Mutual Fund should be the primary investment choice
for Rajesh. It has the highest suitability score of 0.94 because it perfectly
balances his moderate risk profile with his 5-year goal. This fund offers 8-10%
annual returns historically, which is sufficient to help achieve his 2 million
rupee target. Additionally, it suits his Steady Saver persona as it provides
the stability he seeks while maintaining growth potential. I recommend starting
with a 40% allocation to the Balanced Mutual Fund.

User Question: What's the success probability for reaching the goal?

AI Response: The probability of Rajesh reaching his 2 million INR goal is 0.72
or 72%. This is a healthy success probability. He has three key factors working
in his favor: a strong current savings base of 500,000 INR, a reasonable
5-year timeline, and stable income to support monthly investments. However,
success requires consistent monthly investments of approximately 30,000 INR and
assumes average market returns of 8-10%. The main risks are missing monthly
contributions and unexpected market downturns.

EXAMPLE 3: TEST EXECUTION OUTPUT

Running tests.py output:

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
Total Execution Time: 12.34 seconds

TEST REPORT: ALL TESTS PASSED SUCCESSFULLY


EXAMPLE 4: MODEL TRAINING OUTPUT

Running ml/training/train_models.py:

Initializing model training...

Training Risk Profile Model
  Loading data...
  Creating RandomForestClassifier...
  Training on sample data...
  Model accuracy: 0.89
  Saving model to ml/models/risk_profile_model.pkl
  Saving label encoders to ml/models/label_encoders.pkl
  Status: SUCCESS

Training Goal Success Model
  Loading data...
  Creating RandomForestClassifier...
  Training on sample data...
  Model accuracy: 0.82
  Saving model to ml/models/goal_success_model.pkl
  Saving label encoders to ml/models/goal_success_label_encoders.pkl
  Status: SUCCESS

Model Training Completed
  Risk model saved successfully
  Goal model saved successfully
  Encoders saved successfully
  Ready for production use
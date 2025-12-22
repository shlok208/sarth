#!/usr/bin/env python3
"""Script to add execute_leads_insights function"""

import re

file_path = 'backend/agents/tools/Chase_Leads_manager.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Update import to include LeadsInsightsPayload
content = re.sub(
    r'from agents\.emily import LeadsManagementPayload',
    r'from agents.emily import LeadsManagementPayload, LeadsInsightsPayload',
    content
)

# Add execute_leads_insights function before execute_leads_operation
execute_leads_insights_func = '''
def execute_leads_insights(payload: LeadsInsightsPayload, user_id: str, asked_questions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Execute leads insights operation based on the payload
    
    Args:
        payload: LeadsInsightsPayload with insight details
        user_id: User ID for the request
        asked_questions: Dict tracking which questions have been asked
        
    Returns:
        Dict with one of:
        - {"success": False, "clarifying_question": "..."} if missing required fields
        - {"success": False, "error": "..."} if error
        - {"success": True, "data": {...}} if operation succeeds
    """
    try:
        # Initialize asked_questions if not provided
        if asked_questions is None:
            asked_questions = {}
        
        # Extract insight-specific fields
        lead_id = payload.lead_id
        lead_name = payload.lead_name
        platform = payload.platform
        date_range = payload.date_range
        insight_type = payload.insight_type
        
        # If no lead identifier specified, ask for clarification
        if not lead_id and not lead_name:
            question_key = "lead_id_insight"
            question_text = "Which lead would you like insights for? Please provide the lead name or ID."
            
            if question_key in asked_questions:
                return {
                    "success": False,
                    "clarifying_question": f"I already asked which lead you want insights for. Please provide the lead name or ID to continue. Previously asked: '{asked_questions[question_key]}'"
                }
            
            asked_questions[question_key] = question_text
            return {
                "success": False,
                "clarifying_question": question_text,
                "asked_questions": asked_questions
            }
        
        # Build insight query parameters
        lead_identifier = lead_id or lead_name
        platform_text = f" from {platform}" if platform else ""
        date_range_text = f" for {date_range}" if date_range else ""
        insight_type_text = f" ({insight_type})" if insight_type else ""
        
        # TODO: Integrate with actual lead insights database query
        # This would query the leads table and generate insights like:
        # - Lead conversion rate
        # - Lead source performance
        # - Lead status trends
        # - Lead engagement metrics
        # - Platform-specific insights
        
        return {
            "success": True,
            "data": {
                "message": f"I'll generate insights for lead {lead_identifier}{platform_text}{date_range_text}{insight_type_text}. This feature is being set up.",
                "insights": {
                    "lead_id": lead_id,
                    "lead_name": lead_name,
                    "platform": platform,
                    "date_range": date_range,
                    "insight_type": insight_type or "summary",
                    "summary": "Lead insights will include conversion rates, engagement metrics, status trends, and platform performance analysis."
                }
            }
        }
            
    except Exception as e:
        logger.error(f"Error in execute_leads_insights: {e}")
        return {
            "success": False,
            "error": str(e)
        }

'''

# Add before execute_leads_operation
content = re.sub(
    r'(def execute_leads_operation\(payload: LeadsManagementPayload)',
    execute_leads_insights_func + r'\1',
    content
)

# Write back
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated Chase_Leads_manager.py with execute_leads_insights function!")













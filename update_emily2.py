#!/usr/bin/env python3
"""Script to add handle_leads_insights and remove inquire_status references"""

import re

file_path = 'backend/agents/emily.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add handle_leads_insights method after handle_leads_management
handle_leads_insights_method = '''    def handle_leads_insights(self, state: IntentBasedChatbotState) -> IntentBasedChatbotState:
        """Handle leads insights intent"""
        try:
            from agents.tools.Chase_Leads_manager import execute_leads_insights

            # Prefer the unvalidated partial payload
            partial_payload = state.get("partial_payload", {}) or {}
            insights_dict = partial_payload.get("leads_insights")

            if not insights_dict:
                state["response"] = "Which lead would you like insights for? Please provide the lead name or ID."
                state["needs_clarification"] = True
                return state

            try:
                payload = LeadsInsightsPayload(**insights_dict)
            except Exception as e:
                logger.error(f"Failed to validate leads insights payload: {e}")
                state["response"] = "I couldn't understand the leads insights request. Please provide a lead name or ID."
                state["needs_clarification"] = True
                return state

            # Check for missing required fields
            if not payload.lead_id and not payload.lead_name:
                state["response"] = "Which lead would you like insights for? Please provide the lead name or ID."
                state["needs_clarification"] = True
                state["partial_payload"] = partial_payload
                return state

            result = execute_leads_insights(payload, state["user_id"])

            if result.get("clarifying_question"):
                state["response"] = result["clarifying_question"]
                state["needs_clarification"] = True
            elif result.get("success") and result.get("data"):
                state["response"] = self._format_leads_insights_response(result["data"])
            elif result.get("error"):
                state["response"] = f"I encountered an error: {result['error']}"
            else:
                state["response"] = "I've processed your leads insights request."
                
        except Exception as e:
            logger.error(f"Error in handle_leads_insights: {e}")
            state["response"] = "I encountered an error while processing your leads insights request. Please try again."
        
        return state
    
'''

# Find handle_leads_management method and add handle_leads_insights after it
pattern = r'(def handle_leads_management\(self, state: IntentBasedChatbotState\) -> IntentBasedChatbotState:.*?return state\n\n    def handle_posting_manager)'
match = re.search(pattern, content, re.DOTALL)
if match:
    content = content[:match.end()-len('\n\n    def handle_posting_manager')] + handle_leads_insights_method + '\n    def handle_posting_manager' + content[match.end()-len('\n\n    def handle_posting_manager'):]

# 2. Add handle_leads_insights to workflow
content = re.sub(
    r'(workflow\.add_node\("handle_leads_management", self\.handle_leads_management\))\s+(workflow\.add_node\("handle_posting_manager")',
    r'\1\n        workflow.add_node("handle_leads_insights", self.handle_leads_insights)\n        \2',
    content
)

# 3. Add edge for handle_leads_insights
content = re.sub(
    r'(workflow\.add_edge\("handle_leads_management", "generate_final_response"\))\s+(workflow\.add_edge\("handle_posting_manager")',
    r'\1\n        workflow.add_edge("handle_leads_insights", "generate_final_response")\n        \2',
    content
)

# 4. Add leads_insights to route_by_intent mapping
content = re.sub(
    r'("leads_management": "handle_leads_management",)\s+("posting_manager": "handle_posting_manager")',
    r'\1\n                "leads_insights": "handle_leads_insights",\n                \2',
    content
)

# 5. Update classification prompt to include leads_insights
content = re.sub(
    r'(Classify it into the correct intent \(one of: content_generation, analytics, leads_management, posting_manager, general_talks\))',
    r'Classify it into the correct intent (one of: content_generation, analytics, leads_management, leads_insights, posting_manager, general_talks)',
    content
)

# 6. Update heuristic routing to prioritize leads_insights
content = re.sub(
    r'(if any\(keyword in query_lower for keyword in \["lead ", " leads", "leadgen", "lead gen", "lead_id", "lead id", "lead status", "lead insight", "insight"\]\):)',
    r'if any(keyword in query_lower for keyword in ["lead insight", "leads insight", "lead insights", "leads insights", "insight"]):\n            # Prioritize leads_insights for insight-related queries\n            seeded_payload = partial_payload.copy() if isinstance(partial_payload, dict) else {}\n            if "leads_insights" not in seeded_payload:\n                seeded_payload["leads_insights"] = {}\n            state["partial_payload"] = seeded_payload\n            state["intent_payload"] = IntentPayload(\n                intent="leads_insights",\n                content=None,\n                analytics=None,\n                leads=None,\n                leads_insights=None,\n                posting=None,\n                general=None,\n            )\n            logger.info("Heuristic routing to leads_insights based on insight-related keywords.")\n            return state\n        \n        if any(keyword in query_lower for keyword in ["lead ", " leads", "leadgen", "lead gen", "lead_id", "lead id"]):',
    content
)

# 7. Remove all inquire_status and inquire_status_summary references from normalization
content = re.sub(
    r'(\s+"export": "export_leads",)\s+"status": "inquire_status",\s+"status_summary": "inquire_status_summary",\s+"inquire_status": "inquire_status",\s+"inquire_status_summary": "inquire_status_summary",',
    r'\1',
    content
)

# 8. Remove inquire_status from lead_fields list
content = re.sub(
    r'("date_range",)\s+("key",)\s+("status_type",)',
    r'\1\n                \2',
    content
)

# 9. Remove inquire_status inference logic
content = re.sub(
    r'(\s+# If no action yet, infer from nested blocks.*?leads_payload\["action"\] = "inquire_status_summary")',
    r'',
    content,
    flags=re.DOTALL
)

# 10. Remove inquire_status from options in handle_leads_management
content = re.sub(
    r'(\["add_lead", "update_lead", "search_lead", "export_leads",)\s*"inquire_status",\s*"inquire_status_summary"\]',
    r'\1]',
    content
)

# 11. Add _format_leads_insights_response method
format_method = '''    def _format_leads_insights_response(self, data: Dict[str, Any]) -> str:
        """Format leads insights response"""
        if isinstance(data, dict):
            if "message" in data:
                return data["message"]
            elif "insights" in data:
                insights = data["insights"]
                response = f"Here are the insights for lead {insights.get('lead_name') or insights.get('lead_id', 'N/A')}:\n"
                if insights.get("summary"):
                    response += f"\n{insights['summary']}"
                return response
        return str(data)
    
'''
# Add after _format_leads_response
pattern = r'(def _format_leads_response\(self, data: Dict\[str, Any\]\) -> str:.*?return str\(data\)\n\n    def handle_posting_manager)'
match = re.search(pattern, content, re.DOTALL)
if match:
    content = content[:match.end()-len('\n\n    def handle_posting_manager')] + format_method + '\n    def handle_posting_manager' + content[match.end()-len('\n\n    def handle_posting_manager'):]

# Write back
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated emily.py with leads_insights handler and removed inquire_status references!")













#!/usr/bin/env python3
"""Script to update emily.py: remove inquire_status and add leads_insights"""

import re

file_path = 'backend/agents/emily.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove inquire_status and inquire_status_summary from LeadsManagementPayload action Literal
content = re.sub(
    r'(\s+"export_leads",)\s+"inquire_status",\s+"inquire_status_summary"',
    r'\1',
    content
)

# 2. Remove status_type field from LeadsManagementPayload
content = re.sub(
    r'(\s+key: Optional\[str\] = None)\s+status_type: Optional\[str\] = None',
    r'\1',
    content
)

# 3. Add LeadsInsightsPayload class after exportleadPayload
leads_insights_class = '''# =============================================================================
# LEADS INSIGHTS
# =============================================================================

class LeadsInsightsPayload(BaseModel):
    lead_id: Optional[str] = None
    lead_name: Optional[str] = None
    platform: Optional[str] = None
    date_range: Optional[str] = Field(
        default=None,
        description="Example: today, yesterday, last 7 days, this week, last month"
    )
    insight_type: Optional[str] = Field(
        default=None,
        description="Type of insight: summary, conversion, engagement, performance, etc."
    )

'''
content = re.sub(
    r'(class exportleadPayload\(BaseModel\):\s+platform: Optional\[str\] = None\s+date_range: Optional\[str\] = None)\s+',
    r'\1\n' + leads_insights_class,
    content
)

# 4. Add leads_insights to IntentPayload intent Literal
content = re.sub(
    r'("leads_management",)\s+("posting_manager",)',
    r'\1\n        "leads_insights",\n        \2',
    content
)

# 5. Add leads_insights field to IntentPayload
content = re.sub(
    r'(leads: Optional\[LeadsManagementPayload\] = None)\s+(posting: Optional\[PostingManagerPayload\] = None)',
    r'\1\n    leads_insights: Optional[LeadsInsightsPayload] = None\n    \2',
    content
)

# 6. Add LeadsInsightsPayload.model_rebuild()
content = re.sub(
    r'(IntentPayload\.model_rebuild\(\)\s+ContentGenerationPayload\.model_rebuild\(\))',
    r'\1\nLeadsInsightsPayload.model_rebuild()',
    content
)

# Write back
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated emily.py successfully!")













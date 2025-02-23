"""
This script is used to generate a prompt for the GPT-3 model 
to generate a summary of the annual report.
"""

from google import genai

BASE_PROMPT = """
Does the annual report include the following parameters? If the information is available, count 1; otherwise, count 0. Could you give me the total number of counts? Ask me if you need any clarification. Just print the final integer, nothing else.

PARAMETERS
Nitrogen Oxide Emissions 
VOC Emissions 
Carbon Monoxide Emissions 
Particulate Emissions 
Sulphur Dioxide / Sulphur Oxide Emissions
Emissions Reduction Initiatives
Climate Change Policy
Climate Change Opportunities Discussed
Risks of Climate Change Discussed
Direct CO2 Emissions 
Indirect CO2 Emissions 
ODS Emissions 
GHG Scope 1
GHG Scope 2
GHG Scope 3
Scope 2 Market Based GHG Emissions
Scope of Disclosure
Carbon per Unit of Production
Biodiversity Policy
Number of Environmental Fines
Environmental Fines (Amount)
Number of Significant Environmental Fines
Amount of Significant Environmental Fines
Energy Efficiency Policy
Total Energy Consumption
Renewable Energy Use
Electricity Used
Fuel Used - Coal/Lignite
Fuel Used - Natural Gas
Fuel Used - Crude Oil/Diesel
Self Generated Renewable Electricity
Energy Per Unit of Production
Waste Reduction Policy
Hazardous Waste 
Total Waste 
Waste Recycled 
Raw Materials Used 
% Recycled Materials
Waste Sent to Landfills
Percentage Raw Material from Sustainable Sources
Environmental Supply Chain Management
Water Policy
Total Water Discharged
Water per Unit of Production
Total Water Withdrawal
Water Consumption
Human Rights Policy
Policy Against Child Labor
Quality Assurance and Recall Policy
Consumer Data Protection Policy
Community Spending
Number of Customer Complaints
Total Corporate Foundation and Other Giving
Equal Opportunity Policy
Gender Pay Gap Breakout
% Women in Management
% Women in Workforce
% Minorities in Management
% Minorities in Workforce
% Disabled in Workforce
Percentage Gender Pay Gap for Senior Management
Percentage Gender Pay Gap Mid & Other Management
Percentage Gender Pay Gap Employees Ex Management
% Gender Pay Gap Tot Empl Including Management
% Women in Middle and or Other Management
Business Ethics Policy
Anti-Bribery Ethics Policy
Political Donations
Health and Safety Policy
Fatalities - Contractors
Fatalities - Employees
Fatalities - Total
Lost Time Incident Rate
Total Recordable Incident Rate
Lost Time Incident Rate - Contractors
Total Recordable Incident Rate - Contractors
Total Recordable Incident Rate - Workforce
Lost Time Incident Rate - Workforce
Training Policy
Fair Renumeration Policy
Number of Employees - CSR
Employee Turnover %
% Employees Unionized
Employee Training Cost
Total Hours Spent by Firm - Employee Training
Number of Contractors
Social Supply Chain Management
Number of Suppliers Audited
Number of Supplier Audits Conducted
Number Supplier Facilities Audited
Percentage of Suppliers in Non-Compliance
Percentage Suppliers Audited
Audit Committee Meetings
Years Auditor Employed
Size of Audit Committee
Number of Independent Directors on Audit Committee
Audit Committee Meeting Attendance Percentage
Company Conducts Board Evaluations
Size of the Board
Number of Board Meetings for the Year
Board Meeting Attendance %
Number of Executives / Company Managers
Number of Non Executive Directors on Board
Company Has Executive Share Ownership Guidelines
Director Share Ownership Guidelines
Size of Compensation Committee
Num of Independent Directors on Compensation Cmte
Number of Compensation Committee Meetings
Compensation Committee Meeting Attendance %
Number of Independent Directors
Size of Nomination Committee
Num of Independent Directors on Nomination Cmte
Number of Nomination Committee Meetings
Nomination Committee Meeting Attendance Percentage
Verification Type
Employee CSR Training
Board Duration (Years)


ANNUAL REPORT
"""

with open(
    "../parsed/hdfc/HDFC Bank Annual Report 2021-22.pdf.txt", "r", encoding="utf-8"
) as file:
    data = file.read()

client = genai.Client(api_key="AIzaSyDWO0g0pqACnM2jHxbOBu57nA_CfncNb6k")
response = client.models.generate_content(
    model="gemini-2.0-flash", contents=f"{BASE_PROMPT}\n{data}"
)
print(response.text)

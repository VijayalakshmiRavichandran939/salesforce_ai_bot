from simple_salesforce import Salesforce
import os
from dotenv import load_dotenv

load_dotenv()

def connect_salesforce():
    return Salesforce(
        username=os.getenv("SALESFORCE_USERNAME"),
        password=os.getenv("SALESFORCE_PASSWORD"),
        security_token=os.getenv("SALESFORCE_SECURITY_TOKEN"),
        domain=os.getenv("SALESFORCE_DOMAIN")
    )

def create_opportunity(sf, name, stage, close_date, amount):
    return sf.Opportunity.create({
        'Name': name,
        'StageName': stage,
        'CloseDate': close_date,
        'Amount': amount
    })

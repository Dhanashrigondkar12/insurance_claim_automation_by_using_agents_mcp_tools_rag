from fastapi import FastAPI
from pydantic import BaseModel
from agent import graph


#######################################################################################################
#CREATE FASTAPI APPLICATION
#######################################################################################################

app = FastAPI(
    title = "ERFO CLAIM MANAGEMENT API.",
    description = "AI-powered insurance claim processing API.",
    version = "1.0.0"
)

##########################################################################################
#REQUIRED MODEL
############################################################################################

class ClaimRequest(BaseModel):
    claim_id : str
    
    
    
###############################################################################################
#health check
################################################################################################
@app.get("/")
async def root():
    return { 
            "message" : "ERGO Claim management API is runinnng."}


###########################################################################################
##Claim Processing
###########################################################################################


@app.post("/process_claim")
async def process_claim(request : ClaimRequest):
    
    initial_state ={
        
        "claim_id" : request.claim_id,
        "claim" : {},
        "document" : {},
        "validation" : {},
        "policy" : {},
        "fraud" : {},
        "recommendation" : {}
    }
    
    #RUN LANGRAPH
    
    result = await graph.ainvoke(initial_state)

    return {
        
        "claim"  : result["claim"],
        
        "document" : result["document"],
        
        "validation" : result["validation"],
        
        "policy" :  result["policy"],
        
        'fraud' :  result["fraud"],
        
        "recommendation" :  result["recommendation"]
        
        
    
    }
        
        
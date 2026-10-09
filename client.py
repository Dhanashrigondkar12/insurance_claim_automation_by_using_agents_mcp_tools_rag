import asyncio
from fastmcp import Client
from pathlib import Path

async def  main():
    #connect to our MCP Server
    server_path = Path(__file__).parent / "server.py"
    client = Client(server_path)
    
    async with client:
        print("\n Conected to ERGO MCP Swrver\n")
        
        ###Discover available tools
        
        tools = await client.list_tools()
        print('Available MCP tools:')
        
        
        for tool in tools:
            print("-", tool.name)
            
            
            #Intake 
            
        claim = await client.call_tool(
            "intake_claim",
            {
                "claim_id" : "CLM1001"
            }
            
        )
        
        print("\n1. CLAIM INFORMATION.")
        print(claim)
        
        
        #DOCUMENTY EXTRACTION
        document = await client.call_tool(
            "extract_document",{
                "document_text" : """ 
                 Accidnet report:  BMW Vehicle.
                 Accident date : 2026-09-20,
                 Damage amount : 850000
                 Registration : MH1232AA
                
                
                
                
                """
            }
        )
        
        print('\n Document extraction')
        print(document)
        
        #validation
        
        validation = await client.call_tool(
            "validate_claim",
            {
                "claim_amount" : 85000,
                "damage_amount" : 50000
            }
        )
        
        print('\n Validation')
        print(validation)
        
        
        #Policy check
        
        policy = await client.call_tool(
            "check_policy",
            {
                
                "policy_number" : 'POL1001',
                "claim_amount" : 85000
            }
        )
        
        print('\n Policy check')
        print(policy)
        
        #Fraud check
        
        fraud = await client.call_tool(
            
            "check_fraud_risk",
            {
                "claim_amount" : 85000
                
            }
        )
        
        print('\n. FRAUD CHECK')
        print(fraud)
        
        
        ##Recommendation
        
        recommendation = await client.call_tool(
            
            "generate_recommendation",
            
            {
                "validation_passed" : True,
                "policy_covered" : True,
                "fraud_risk_level" :"LOW"
            }
        )
        
        print('\n6. FINAL RECOMMENDATION')
        print(recommendation)
        
        
        
if __name__ == '__main__':
    asyncio.run(main())
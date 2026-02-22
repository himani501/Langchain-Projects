from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "llama3.2"

# ----- Tools ----- 

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f" >> Executing get_product_price(product= '{product}')")
    prices = {"laptop": 1299.99, "headphones": 89.00, "keyboard": "79.65"}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f" >> Executing apply_discount on product of price={price} and discount_tier={discount_tier}")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1-discount/100), 2)

@traceable(name="Langchain agent loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    message = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to product catalog tool "
                "and a discount tool.\n\n"
                "WORKFLOW - Follow this exact sequence:\n"
                "Step 1: If user asks for a product price, call get_product_price with the product name as a string\n"
                "Step 2: If user mentions ANY discount (bronze/silver/gold), you MUST call apply_discount with the price from Step 1\n"
                "Step 3: Return the final discounted price to the user\n\n"
                "EXAMPLE:\n"
                "User: 'What is the price of laptop after applying gold discount?'\n"
                "You must: 1) Call get_product_price('laptop') 2) Call apply_discount(price_from_step1, 'gold') 3) Return final price\n\n"
                "STRICT RULES:\n"
                "- NEVER skip the apply_discount tool when discount is mentioned\n"
                "- NEVER calculate discounts manually\n"
                "- Pass product names as simple strings, not dictionaries\n"
                "- Available discount tiers: bronze, silver, gold"
            )
        ),
        HumanMessage(content=question)
    ]

    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"\n--- Iteration {iteration} ---")
        ai_message = llm_with_tools.invoke(message)

        tool_calls = ai_message.tool_calls

        #if no tool call, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
        
        #Process only the first tool call
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"[Tool selected]: {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        try:
            observation = tool_to_use.invoke(tool_args)
        except Exception as e:
            print(f" [Tool Error]: {e}")
            # Send error back to the LLM so it can try again with correct parameters
            observation = f"Error: {str(e)}. Please check your parameters and try again."

        print(f" [Tool Result]: {observation}")

        message.append(ai_message) #feeding llm with the result again
        message.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )   #feeding the tool message as well back to llm

    print("Error: Max iterations reached without final answer!")
    return None



if __name__ == "__main__":
    print("Hello Langchain Agent")
    print()
    result = run_agent("What is the price of laptop after applying gold discount?")

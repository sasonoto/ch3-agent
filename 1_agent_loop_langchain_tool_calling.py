from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from langsmith import traceable

MAX_ITERATIONS = 3
MODEL = "gemini-3-flash-preview"

# --- Tools (LangChain @tool decorated functions) ---

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {
        "laptop": 1299.99,
        "headphones": 199.99,
        "keyboard": 89.99
    }
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price and return the final price.
    Available discount tiers are: 'bronze', 'silver', 'gold'."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {
        "bronze": 15,
        "silver": 30,
        "gold": 40
    }
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount/100), 2)

# --- Agent Loop ---
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(f"google_genai:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 61 )

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant that answers questions about product prices and discounts. "
                "You have access to the following tools: get_product_price and apply_discount. "
                "Use these tools to find the price of products and apply discounts as needed."
            )
        ),
        HumanMessage(content=question)
    ]

    for iteration in range(MAX_ITERATIONS):
        print(f"Iteration {iteration + 1}:")
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        # If no tool calls, we assume the model is giving a final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
        
        # Process only the first tool call for simplicity
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"  Model called tool: {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found in tools_dict.")    
        
            print(f"  Error: Tool '{tool_name}' not found.")
            return f"Error: Tool '{tool_name}' not found."
        
        observation = tool_to_use.invoke(tool_args)

        print(f"   [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call_id
            )
        )

    print("\nMaximum iterations reached without a final answer.")
    return None

if __name__ == "__main__":
    print("Welcome to the Agent Loop with LangChain Tools (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop with a gold discount?")
    print(result)

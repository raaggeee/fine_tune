import json

def modify_data(data_path):
    final_data = []
    with open(data_path, "r") as f:
        data = json.load(f)

    for i in data:
        tool_desc = i["NLDocumentation"]
        for inst in i["Instances"]:
            try:
                tool_steps = inst["intermediate_steps"][0]
                final_data.append({
                    "input": inst["input"],
                    "output": inst["output"],
                    "tool_name": tool_steps[0][0],
                    "tool_input": tool_steps[0][1],
                    "tool_reasoning": tool_steps[0][2],
                    "tool_result": tool_steps[1],
                    "tool_desc": tool_desc
                })

            except:
                print("\nNot found\n")

    with open("fine_tune_tool/final_data.json", "w") as f:
        json.dump(final_data, f, indent=3)

modify_data("fine_tune_tool/train_data.json")

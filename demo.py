from voyager import Voyager

# You can also use mc_port instead of azure_login, but azure_login is highly recommended
# mc_port is only for local testing and will not work in the cloud.
azure_login = {
    "client_id": "3ba45205-c9c0-46bc-8216-4b481f55873d",
    "redirect_url": "https://127.0.0.1/auth-response",
    "secret_value": "",
    "version": "fabric-loader-0.14.18-1.19", # the version Voyager is tested on
}
openai_api_key = ""

voyager = Voyager(
    mc_port="55612",
    azure_login=azure_login,
    #openai_api_key="sk-fy9hhxeIQoPzAQw8tVJuw0Q1bvx08WBXJNtMY-RpS1T3BlbkFJQJY-gQrxgKxBcnR9kZWRzdGuh0rT2haza0Xdt1fTQA",
    openai_api_key= "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",
    #ANTHROPIC_API_KEY="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",
    #skill_library_dir="/Users/daisysong/Desktop/CS194agent/Voyager_OAI/skill_library/trial1", # Load a learned skill library.
    #ckpt_dir="/Users/daisysong/Desktop/Voyager2/checkpoints", # Feel free to use a new dir. Do not use the same dir as skill library because new events will still be recorded to ckpt_dir. 
    resume = True,
)

# start lifelong learning
voyager.learn()
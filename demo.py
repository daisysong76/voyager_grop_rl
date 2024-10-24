from voyager import Voyager

# You can also use mc_port instead of azure_login, but azure_login is highly recommended
azure_login = {
    "client_id": "3ba45205-c9c0-46bc-8216-4b481f55873d",
    "redirect_url": "https://127.0.0.1/auth-response",
    "secret_value": "",
    "version": "fabric-loader-0.14.18-1.19", # the version Voyager is tested on
}
openai_api_key = "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"

voyager = Voyager(
    mc_port="50057",
    openai_api_key=openai_api_key,
    resume = True,
)

# start lifelong learning
voyager.learn()
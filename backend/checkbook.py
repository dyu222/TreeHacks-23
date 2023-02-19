import requests

# Define the endpoint and request parameters
def create_invoice(request):
    endpoint = 'https://api.checkbook.io/v3/checks'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'YOUR_API_KEY'
    }
    # will want to customize this to our needs
    data = {
        'recipient': 'John Doe',
        'amount': 100.0,
        'description': 'Payment for services',
        'email': 'john.doe@example.com'
    }

    # Make the API call
    response = requests.post(endpoint, headers=headers, json=data)

    # Check the response status code
    if response.status_code == 200:
        print('API call successful')
    else:
        print(f'API call failed with status code {response.status_code}')

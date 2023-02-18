from flask import Flask, request, jsonify
from caption_recommendation import recommend_land


app = Flask(__name__)

@app.route('/recommendation', methods=['POST'])
def recommendation():
    user_input = request.json.get('input')
    user_min_price = float(request.json.get('min_price'))
    user_max_price = float(request.json.get('max_price'))
    user_min_acres = float(request.json.get('min_acres'))
    user_max_acres = float(request.json.get('max_acres'))

    result = recommend_land(user_input, user_min_price, user_max_price, user_min_acres, user_max_acres)

    return result

@app.route('/create_invoice', methods=['POST'])
def create_invoice():
    api_key = 'YOUR_API_KEY'  # Replace with your Checkbook API key
    endpoint = 'https://api.checkbook.io/v3/invoices'
    # Extract data from the request
    data = request.get_json()
    amount = data.get('amount')
    recipient = data.get('recipient')
    email = data.get('email')
    description = data.get('description')
    # Create the request data
    request_data = {
        'recipient': recipient,
        'amount': amount,
        'email': email,
        'description': description
    }
    # Make the API call
    headers = {
        'Content-Type': 'application/json',
        'Authorization': api_key
    }
    response = request.post(endpoint, headers=headers, json=request_data)
    # Check the response status code
    if response.status_code == 200:
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': response.text}), 400

if __name__ == '__main__':
    app.run(debug=True)
# # we can alter the / to change what api path we want in react 
# @api.route('/profile')
# def my_profile():
#     # edit these to call our code and host it as an api
#     response_body = {
#         "name": "Nagato",
#         "about" :"Hello! I'm a full stack developer that loves python and javascript"
#     }

#     return response_body
from flask import Flask, request
from backend.caption_recommendation import recommend_land


app = Flask(__name__)

@app.route('/recommendation', methods=['POST'])
def recommendation():
    user_input = request.form.get('input')
    user_min_price = float(request.form.get('min_price'))
    user_max_price = float(request.form.get('max_price'))
    user_min_acres = float(request.form.get('min_acres'))
    user_max_acres = float(request.form.get('max_acres'))

    result = recommend_land(user_input, user_min_price, user_max_price, user_min_acres, user_max_acres)

    return result

# # we can alter the / to change what api path we want in react 
# @api.route('/profile')
# def my_profile():
#     # edit these to call our code and host it as an api
#     response_body = {
#         "name": "Nagato",
#         "about" :"Hello! I'm a full stack developer that loves python and javascript"
#     }

#     return response_body
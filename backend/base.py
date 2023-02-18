from flask import Flask

api = Flask(__name__)

# we can alter the / to change what api path we want in react 
@api.route('/profile')
def my_profile():
    # edit these to call our code and host it as an api
    response_body = {
        "name": "Nagato",
        "about" :"Hello! I'm a full stack developer that loves python and javascript"
    }

    return response_body
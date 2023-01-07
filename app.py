from flask import Flask, redirect, url_for, render_template, request, flash

import module1 as m1
# import module2 as m2
# import module3 as m3

app = Flask(__name__)
app.secret_key = "hello"

@app.route("/",  methods=["POST", "GET"])
def home():
    if (request.method == "GET"):
        return render_template("index.html")
    else:
        input_image_file = request.files['input_image_file']
        input_model_selection = request.form['input_model_selection']

        # Save the uploaded image in static/
        input_image_path = './static/' + input_image_file.filename
        input_image_file.save(input_image_path)

        model_name = ""
        # flash("modal=" + input_model_selection)
        # if (input_model_selection == 1):
            # flash("BOOGA")
        if (input_model_selection == "1"):
            #  then UNet was picked
            # outputs format: imagename_modelname_MASK.jpg
            # will be in static/, of course
            model_name = "UNet"
            output_image_path = m1.predictAndSaveOutputFor(input_image_file.filename)

            return render_template("prediction.html", mask_load_path=output_image_path, input_load_path=input_image_path)
        else:
            return "not model 1 nygga"
        # return render_template("index.html"); 


if __name__ == "__main__":
    app.run(debug=True); 


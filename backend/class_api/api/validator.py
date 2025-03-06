from marshmallow import Schema, fields, validate, ValidationError
import jsonify

class CarDetailSchema(Schema):
    Brand  = fields.Str()
    Model = fields.Str()
    Year  = fields.Integer()
    Engine_Size  = fields.Float()
    Fuel_Type = fields.Str()
    Transmission = fields.Str()
    Mileage = fields.Integer()
    Doors = fields.Integer() 
    Owner_Count = fields.Integer()


    def validate_input(self, input_form):
        
        schema = CarDetailSchema( many=True)
        try:
            validated_data = schema.load(input_form)
            return validated_data
        except ValidationError as err:
            return jsonify({'message': err}), 400
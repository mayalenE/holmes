        var class_id_map = [];
        var show_from = 5000;

        
        $(document).ready(function(){

            class_id_map["All"]=[];
            for(i=0;i<5000;i++) class_id_map["All"].push(i);

            for(i = 48; i > 0; i--){
                $("#show_result").append(
                    "<div class='lenia_img' id='img_"+ (i-1) + "' style='display:inline-block;text-align:center'>"+
                        "<span style='font-size:11px'></span><br>"+
                        "<img  src='' >"+
                    "</div>"
                );
            }
        })
        $(document).ready(function(){
            $.getJSON( "https://raw.githubusercontent.com/mayalenE/holmes/gh-pages/assets/media/experiments/experiment_names.json", function( data ) {
                var items = [];
                $.each( data, function( key, val ) {
                    $("#experiment").append( "<option value='" + val + "'>" + val + "</option>" );
                });
                displayImagesWithClassFromNumber(5000, class_id_map["All"]);
            });


            $("#load_images").submit(function(){
                displayImagesWithClassFromNumber(show_from, class_id_map["All"]);

                $.getJSON("https://raw.githubusercontent.com/mayalenE/holmes/gh-pages/assets/media/experiments/"+ $("#experiment").val() +  "/repetition_" + pad( $("#repetition").val(),6) +"/classes.json", function( data ) {
                    $("#classes").html('<option value="All">All</option>');
                    $.each( data, function( key, val ) {
                        $("#classes").append( "<option value='" + key + "'>" + key + "</option>" );
                        class_id_map[key]= val;
                    });
                    $("#window_slider").val("5000")
                });

                return false;
            })

            $("#classes").change(function(){
                var index_window = $("#window_slider").val();
                displayImagesWithClassFromNumber(index_window,class_id_map[$("#classes").val()]);
            })  
            $("#window_slider").change(function(){
                var index_window = $("#window_slider").val();
                displayImagesWithClassFromNumber(index_window, class_id_map[$("#classes").val()]);
            })

        })


        // start from
        function displayImagesWithClassFromNumber(start_from_image, image_map_for_class){
            // index of the image in the map
            index_in_map = image_map_for_class.indexOf(start_from_image);

            // if not found returns ind<0
            i = start_from_image;
            while(index_in_map < 0 && i >= 48){
                index_in_map = image_map_for_class.indexOf(i--);
            }

            index_in_map++

            // if found the image but there is less then 48 images before it
            // take the first 48 images quand meme
            if(index_in_map < 48) index_in_map= 48;

            //show images
            for(i = 1; i <= 48; i++){
                var image_number = image_map_for_class[index_in_map - (i)];
                $("#img_"+(i-1)+" img").attr('src',"https://raw.githubusercontent.com/mayalenE/holmes/gh-pages/assets/media/experiments/"+ $("#experiment").val() +  "/repetition_" + pad( $("#repetition").val(),6) +"/images/" +  (image_number)  +".png");
                $("#img_"+(i-1)+" span").html( image_number + 1);
            }
        }


        function pad (str, max) {
            str = str.toString();
            return str.length < max ? pad("0" + str, max) : str;
        }

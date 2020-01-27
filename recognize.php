<!doctype html>

<html lang="en">
<head>
  <link rel="shortcut icon" href="images/favicon.png">
  <link href="https://fonts.googleapis.com/css?family=Montserrat:400,600" rel="stylesheet">
  <link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons">
  <link href="https://fonts.googleapis.com/css?family=Open+Sans:600" rel="stylesheet">
  <link rel="stylesheet" href="https://code.getmdl.io/1.3.0/material.brown-orange.min.css" />
  <!--<link rel="stylesheet" href="https://code.getmdl.io/1.3.0/material.deep_orange-orange.min.css" />-->
  <link rel="stylesheet" href="styles.css">
  <script defer src="https://code.getmdl.io/1.3.0/material.min.js"></script>
  <script type="text/javascript" src="jquery.js"></script>
  <script type="text/javascript" src="jquery.form.js"></script>
  <script>
    $(document).ready(function() 
    {
      var i;
      for (i = 0; i < 3; i++)
      { 
        $('#image_preview1').append("<img src='images/placeholder.png' height = 100px style='margin:10px;' class = 'mdl-shadow--2dp'>");
      } 
      for (i = 0; i < 3; i++)
      { 
        $('#image_preview2').append("<img src='images/placeholder.png' height = 100px style='margin:10px;' class = 'mdl-shadow--2dp'>");
      }  
    });
    function reset1(divName1,divName2)
    {
      $('#image_preview1').empty();
      var i;
      for (i = 0; i < 3; i++)
      { 
        $('#image_preview1').append("<img src='images/placeholder.png' height = 100px style='margin:10px;' class = 'mdl-shadow--2dp'>");
      }
      if( document.getElementById("upload_file1").files.length == 0 ){
        document.getElementById(divName2).style.visibility = 'visible';
//alert(divName2);
}
else
{
  document.getElementById(divName1).style.visibility = 'visible';
}


}
function reset2(divName1,divName2)
{
  $('#image_preview2').empty();
  var i;
  for (i = 0; i < 3; i++)
  { 
    $('#image_preview2').append("<img src='images/placeholder.png' height = 100px style='margin:10px;' class = 'mdl-shadow--2dp'>");
  } 
  if( document.getElementById("upload_file2").files.length == 0 ){
    document.getElementById(divName2).style.visibility = 'visible';
//alert(divName2);
}
else
{
  document.getElementById(divName1).style.visibility = 'visible';
}

}
function preview_image1() 
{
  var total_file=document.getElementById("upload_file1").files.length;
  $('#image_preview1').empty();
  for(var i=0;i<total_file;i++)
  {
    $('#image_preview1').append("<img src='"+URL.createObjectURL(event.target.files[i])+"' height = 150px style='margin-right:20px'>");
  }
}
function preview_image2() 
{
  var total_file=document.getElementById("upload_file2").files.length;
  $('#image_preview2').empty();
  for(var i=0;i<total_file;i++)
  {
    $('#image_preview2').append("<img src='"+URL.createObjectURL(event.target.files[i])+"' height = 150px style='margin-right:20px'>");
  }
}
</script>
<style>
  .black-border
  {
    border: 1px solid black;
  }
  .custom-card{
    padding:20px;
    background: #FFFFFF;
    width: 100%;

  }
  #view-source {
    position: fixed;
    display: block;
    right: 0;
    bottom: 0;
    margin-right: 40px;
    margin-bottom: 40px;
    z-index: 900;
  }
  body
  {
    font-family: 'Montserrat', sans-serif;
  }
  label.input-custom-file input[type=file] 
  {
    display: none;  
  }
  label
  {
    margin-top: 5px;
  }
  .display-1
  { 
    font-size: 28px;
  }
  .display-2
  {
    font-size: 16px;
  }
  .error
  {
    margin-top: 10px;
    color: #ff0000;
  }
</style>
</head>
<body>
  <div class="demo-layout mdl-layout mdl-js-layout mdl-layout--fixed-drawer mdl-layout--fixed-header">
    <header class="demo-header mdl-layout__header mdl-color--grey-100 mdl-color-text--grey-600">
      <div class="mdl-layout__header-row">
        <span class="mdl-layout-title display-1">Recognize</span>
        <div class="mdl-layout-spacer"></div>
      </div>
    </header>
    <div class="demo-drawer mdl-layout__drawer mdl-color--blue-grey-900 mdl-color-text--blue-grey-50">
      <header class="demo-drawer-header">
        <img src="images/aid321678-v4-900px-Analyze-Handwriting-(Graphology)-Step-10.jpg" class="demo-avatar">
        <div class="demo-avatar-dropdown">
          <span class="mdl-color-text--blue-grey-50 display-2" style="margin-top: 10px;">Neural Forensics</span>
        </div>
      </header>
      <nav class="demo-navigation mdl-navigation mdl-color--blue-grey-800">
        <a class="mdl-navigation__link display-2" href="index.html"><i class="mdl-color-text--blue-grey-400 material-icons" role="presentation">home</i>Home</a>              
        <a class="mdl-navigation__link display-2" href="recognize.php"><i class="mdl-color-text--blue-grey-400 material-icons" role="presentation">live_help</i>Recognize</a>
        <a class="mdl-navigation__link display-2" href="match.php"><i class="mdl-color-text--blue-grey-400 material-icons" role="presentation">file_copy</i>Match</a>
        <div class="mdl-layout-spacer"></div>
        <a class="mdl-navigation__link" href=""><i class="mdl-color-text--blue-grey-400 material-icons" role="presentation">help_outline</i><span class="visuallyhidden">Help</span></a>
      </nav>
    </div>
    <main class="mdl-layout__content mdl-color--grey-100">
      <div class="mdl-grid demo-content">

        <!--YOU HAVE TO WORK HERE-->
        <div class="custom-card mdl-shadow--2dp mdl-cell mdl-cell--12-col mdl-grid display-1"> 1. Upload the unknown handwriting samples.</div> 
        <div class="custom-card mdl-shadow--2dp  mdl-cell ">

          <!--YOU HAVE TO MAKE THIS DIV VISIBLE BEFORE SENDING DATA TO PHP AND HIDDEN ONCE RESULT COMES BACK-->
          <div id="train_progress" class="mdl-progress mdl-js-progress mdl-progress__indeterminate" style="border: 0px solid black;width:100%;visibility: hidden;"></div> 

          <div id="image_preview1" style = "overflow-x: auto;white-space: nowrap;width: 100%;margin-bottom: 10px;"></div>
          <form action="recognizesuspect.php" method="post" enctype="multipart/form-data" id = "f1">
            <Label class="input-custom-file mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" style="margin-right: 10px">ADD FILE
              <input type="file" id="upload_file1" name="upload_file1[]" onchange="preview_image1();" multiple required />
            </Label>
            <label class="input-custom-file mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" onclick = "reset1()">RESET</label>
            <input type = "submit" name = "submit_unknown" id = "submit_unknown" value =  "Upload and start the training" class="input-custom-file mdl-button mdl-button--accent mdl-js-button mdl-button--raised mdl-js-ripple-effect mdl-button--colored" onclick = 'reset1("train_progress","upload-error1")' style="float: right;">
          </form>
          <div id = "upload-error1" class = "error" style="visibility: hidden;">Select at least 1 file before uploading</div>
          <!--YOUR WORK ENDS HERE-->


        </div>
        <!-- Separator-->
        


        <?php 
        if (isset($_REQUEST['diff']))
        {
          if(strlen($_REQUEST['diff']) <= 2)
          {
            echo '<div class="custom-card mdl-shadow--2dp display-1">';
            $file = "data.txt";

            $json = json_decode(file_get_contents($file));
            if((int)$_REQUEST['diff'] < 15)
            {
              echo '<div style="font-size:20px;margin-bottom:10px;text-align:center;">No precise matches were found. However, these were some closest matches:</div>';
            }
            echo '<center><table class="mdl-data-table mdl-js-data-table mdl-shadow--2dp">
            <thead>
              <tr>
                <th class="mdl-data-table__cell--non-numeric">Suspect Name</th>
                <th>Probability</th>
              </tr>
            </thead>
            <tbody>';
              for($i = 0 ; $i<count($json->Suspects) ; $i++)
              {

                echo '<tr><td class="mdl-data-table__cell--non-numeric">'.$json->Suspects[$i]->Name.'</td>
                <td>'.(int)$json->Suspects[$i]->Probability.'%</td></tr>';
              }        
              echo '</tbody></table></center>';

              echo '</div>';
            }
          }?>

          <div class="demo-cards mdl-cell mdl-cell--4-col mdl-cell--8-col-tablet mdl-grid mdl-grid--no-spacing">
            <div class="demo-separator mdl-cell--1-col"></div>
          </div>

        </div>
      </main>
    </div>
    <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" version="1.1" style="position: fixed; left: -1000px; height: -1000px;">
      <defs>
        <g id="piechart">
          <circle cx=0.5 cy=0.5 r=0.5 fill = "#3E2723"/>
        </g>
      </defs>
    </svg>
  </body>
  </html>
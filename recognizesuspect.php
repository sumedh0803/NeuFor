<?php
$target_dir = "Recognize/suspect_images/";
#print_r($_FILES["upload_file"]["name"]);
for($i = 0;$i<sizeof($_FILES["upload_file1"]["name"]);$i++)
{
$target_file = $target_dir . basename($_FILES["upload_file1"]["name"][$i]);
$uploadOk = 1;
$imageFileType = strtolower(pathinfo($target_file,PATHINFO_EXTENSION));
// Check if image file is a actual image or fake image
move_uploaded_file($_FILES["upload_file1"]["tmp_name"][$i], $target_file);
}
exec("D:");
exec("cd xampp\\htdocs\\beproj");
$op = exec("E:\\Anaconda3\\python.exe D:\\xampp\\htdocs\\NeuFor\\Recognize.py 2>&1"); 
var_dump($op);              
if(strlen($op)<=2)
{
	$loc = 'recognize.php?diff='.$op;
    echo "<script>window.location.href = '".$loc."';</script>";
}
//redundant code
else
{
	$loc = 'recognize.php?diff='.$op;
    echo "<script>window.location.href = '".$loc."';</script>";
}



?>
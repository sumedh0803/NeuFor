<?php
$target_dir = "Match/unknown_samples/";
    #print_r($_FILES["upload_file"]["name"]);
    for($i = 0;$i<sizeof($_FILES["upload_file2"]["name"]);$i++)
    {
        $target_file = $target_dir . basename($_FILES["upload_file2"]["name"][$i]);
        $uploadOk = 1;
        $imageFileType = strtolower(pathinfo($target_file,PATHINFO_EXTENSION));
        // Check if image file is a actual image or fake image
        move_uploaded_file($_FILES["upload_file2"]["tmp_name"][$i], $target_file);
    }
    exec("D:");
    exec("cd xampp\\htdocs\\beproj");
    $op = exec("E:\\Anaconda3\\python.exe D:\\xampp\\htdocs\\NeuFor\\Match_Test.py 2>&1");
    $op = ((float)$op)*100;
    $loc = 'match.php?percent='.(int)$op;
    //var_dump($op);
    echo "<script>window.location.href = '".$loc."';</script>";

  ?>
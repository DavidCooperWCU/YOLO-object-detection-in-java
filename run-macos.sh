

export PATH_TO_FX=${HOME}/.sdkman/candidates/java/current/jmods

echo Using Path to FX: $PATH_TO_FX

export LD_LIBRARY_PATH=libs:${HOME}/.sdkman/candidates/java/current/lib/jli:/usr/local/share/zbar/lib:$LD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=libs:${HOME}/.sdkman/candidates/java/current/lib/jli:/usr/local/share/zbar/lib:$LD_LIBRARY_PATH

. ${HOME}/.sdkman/bin/sdkman-init.sh



java -Xms1024m -Djava.library.path=$DYLD_LIBRARY_PATH -Djava.io.tmpdir="tmp" -Djdk.gtk.version=2 -Dprism.order=sw -cp build/libs/YOLO-object-detection-in-java-1.0-YOLOv8.jar --module-path $PATH_TO_FX --add-modules javafx.controls,javafx.fxml,javafx.base com.cooplogic.app.VideoYolo

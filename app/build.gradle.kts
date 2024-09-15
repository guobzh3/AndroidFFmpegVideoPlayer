plugins {
    alias(libs.plugins.androidApplication)
}

android {
    namespace = "com.example.ffmpegvideoplayer"
    compileSdk = 29

    defaultConfig {
        applicationId = "com.example.ffmpegvideoplayer"
        minSdk = 20
        targetSdk = 29
        versionCode = 1
        versionName = "1.0"
//        加入renderscript 相关的依赖项目
//        renderscriptTargetApi = 24
//        renderscriptSupportModeEnabled = true
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        externalNativeBuild {
            cmake {
                cppFlags += ""
                abiFilters += "arm64-v8a"
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
//    本地c++链接库
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    buildFeatures {
        mlModelBinding = true
    }
    sourceSets {
        getByName("main") {
            renderscript {
                srcDirs("src\\main\\rs", "src\\main\\rs", "src\\main\\rs", "src\\main\\rs",
                    "src\\main\\rs",
                    "src\\main\\rs"
                )
            }
        }
    }
}

dependencies {

    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.activity)
    implementation(libs.constraintlayout)
    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)

    implementation("org.tensorflow:tensorflow-lite:2.8.0")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.8.0")
    implementation("org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.1")
    implementation("org.tensorflow:tensorflow-lite-support:0.3.1")
    implementation("org.tensorflow:tensorflow-lite-metadata:0.3.1")
}
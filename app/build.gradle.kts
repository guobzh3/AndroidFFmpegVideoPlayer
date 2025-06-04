plugins {
    alias(libs.plugins.androidApplication)
}

android {
    namespace = "com.example.ffmpegvideoplayer"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.ffmpegvideoplayer"
        minSdk = 29
        //noinspection ExpiredTargetSdkVersion
        targetSdk = 34
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
    packagingOptions {
        // ... 可能有上面的 exclude 配置 ...
        jniLibs.useLegacyPackaging = true // 注意 Kotlin DSL 的写法
    }
}

dependencies {

    implementation ("androidx.core:core-ktx:1.12.0")
    implementation(libs.androidx.core.ktx)
    implementation(libs.kotlin.stdlib) // 添加显式的 kotlin-stdlib 依赖

    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.activity)
    implementation(libs.constraintlayout)
    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)

//    implementation("org.tensorflow:tensorflow-lite:2.8.0")
//    implementation("org.tensorflow:tensorflow-lite-gpu:2.8.0")
//    implementation("org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.1")
//    implementation("org.tensorflow:tensorflow-lite-support:0.3.1")
//    implementation("org.tensorflow:tensorflow-lite-metadata:0.3.1")

    implementation("com.google.ai.edge.litert:litert:1.3.0")
    implementation("com.google.ai.edge.litert:litert-gpu:1.3.0")
    implementation("com.google.ai.edge.litert:litert-metadata:1.3.0")
    implementation("com.google.ai.edge.litert:litert-support:1.3.0")


    implementation("com.qualcomm.qti:qnn-litert-delegate:2.34.0")
    implementation("com.qualcomm.qti:qnn-runtime:2.34.0")
}

// 添加 resolutionStrategy 来强制 Kotlin 版本
configurations.all {
    resolutionStrategy {
        force(libs.kotlin.stdlib) // 强制 kotlin-stdlib 的版本
        // 针对 jdk7 和 jdk8 变体，如果它们也被其他依赖引入，也需要强制
        // 通常情况下，强制主模块 kotlin-stdlib 应该能间接影响其变体
        // 但为了更明确，可以分别添加
        eachDependency {
            if (requested.group == "org.jetbrains.kotlin" && requested.name.startsWith("kotlin-stdlib")) {
                useVersion(libs.versions.kotlinStdlib.get())
            }
        }
    }
}
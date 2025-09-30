<script setup>
import { ref } from 'vue'

// Состояние для файла
const file = ref(null)

// Функция загрузки
function handleFileChange(event) {
  const selectedFile = event.target.files[0]
  if (selectedFile && selectedFile.name.endsWith('.zip')) {
    file.value = selectedFile
    // Здесь можно добавить распаковку ZIP и чтение DICOM
    console.log('Загружен файл:', file.value)
  } else {
    alert('Пожалуйста, выберите ZIP файл')
  }
}
</script>

<template>
  <label class="dicom-viewer border border-gray-300 p-6 rounded-lg shadow-lg flex flex-col items-center justify-center">
    
    <img src="/upload-icon.svg" alt="Upload" class="upload-icon" />

    <div class="cursor-pointer bg-[rgba(255,255,255,0.65)] rounded-lg shadow-lg mx-auto text-center px-16 py-10">
      <span class="text-gray-600 text-[20px]">
        Загрузите ZIP файл с DICOM-снимками<br>или перетащите сюда
      </span>

      <input type="file" class="hidden" @change="handleFileChange" accept=".zip" />


      <div v-if="file" class="mt-4 text-center text-gray-700">
        Файл выбран: {{ file.name }}
      </div>
    </div>
  </label>
</template>

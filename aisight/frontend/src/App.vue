<template>
  <div class="p-4">
    <h1 class="text-xl font-bold mb-4">DICOM Viewer</h1>

    <!-- Загрузка файла -->
    <input type="file" @change="onFileChange" accept=".dcm" class="mb-4" />

    <!-- Контейнер для отображения изображения -->
    <div
      ref="dicomElement"
      class="w-[512px] h-[512px] border border-gray-300 relative"
    ></div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

// Cornerstone и утилиты
import cornerstone from 'cornerstone-core'
import dicomParser from 'dicom-parser'
import cornerstoneWADOImageLoader from 'cornerstone-wado-image-loader'

const dicomElement = ref(null)

// Настройка Cornerstone WADO Loader
cornerstoneWADOImageLoader.external.cornerstone = cornerstone
cornerstoneWADOImageLoader.configure({
  beforeSend: function(xhr) {
    // Можно добавить заголовки, если нужно
  },
})

// Функция обработки выбора файла
const onFileChange = (event) => {
  const file = event.target.files[0]
  if (!file) return

  const imageId = cornerstoneWADOImageLoader.wadouri.fileManager.add(file)
  cornerstone.loadImage(imageId).then((image) => {
    cornerstone.displayImage(dicomElement.value, image)
  })
}

// Включаем элемент для Cornerstone после монтирования компонента
onMounted(() => {
  cornerstone.enable(dicomElement.value)
})
</script>

<style scoped>
/* Немного стилей для контейнера */
</style>

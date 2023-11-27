#include "TrainingSchedule.h"

#include "ui_TrainingSchedule.h"

TrainingSchedule::TrainingSchedule(s21::SettingsController& s_controller,
                                   s21::MLPController& m_controller,
                                   QWidget* parent)
    : QWidget(parent),
      ui(new Ui::TrainingSchedule),
      s_controller_(s_controller),
      m_controller_(m_controller) {
  ui->setupUi(this);
  auto abobas = m_controller_.GetMLPsInfo();
  for (const auto& s : abobas) {
    ui->AbobaView->addItem(s.c_str());
  }
  connect(ui->SaveBest, &QCheckBox::clicked, this,
          [&](bool state) { s_controller_.SetSave(state); });
  connect(ui->SaveLog, &QCheckBox::clicked, this,
          [&](bool state) { s_controller_.SetSaveLog(state); });
  connect(ui->BatchSize, &QSpinBox::valueChanged, this, [&](int val) {
    try {
      s_controller_.SetTrainBatchSize(GetMLPIndex(), val);
    } catch (const std::exception& e) {
      ui->ErrorWindow->setText(QString(e.what()));
    }
  });
  connect(ui->Epochs, &QSpinBox::valueChanged, this, [&](int val) {
    try {
      s_controller_.SetEpochs(GetMLPIndex(), val);
    } catch (const std::exception& e) {
      ui->ErrorWindow->setText(QString(e.what()));
    }
  });
  connect(ui->LearningRate, &QDoubleSpinBox::valueChanged, this,
          [&](double val) {
            try {
              s_controller_.SetLearningRate(GetMLPIndex(), val);
            } catch (const std::exception& e) {
              ui->ErrorWindow->setText(QString(e.what()));
            }
          });
  connect(ui->LearningRateReduction, &QDoubleSpinBox::valueChanged, this,
          [&](double val) {
            try {
              s_controller_.SetLearningRateReduction(GetMLPIndex(), val);
            } catch (const std::exception& e) {
              ui->ErrorWindow->setText(QString(e.what()));
            }
          });
  connect(
      ui->LearningRateFrequency, &QSpinBox::valueChanged, this, [&](int val) {
        try {
          s_controller_.SetLearningRateReductionFrequency(GetMLPIndex(), val);
        } catch (const std::exception& e) {
          ui->ErrorWindow->setText(QString(e.what()));
        }
      });
}

TrainingSchedule::~TrainingSchedule() { delete ui; }

size_t TrainingSchedule::GetMLPIndex() {
  if (ui->AbobaView->currentItem()) {
    size_t index = ui->AbobaView->row(ui->AbobaView->currentItem());
    return index;
  } else {
    throw std::range_error("No MLP found");
  }
}
